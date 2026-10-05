"""Các quy tắc dùng CHUNG cho ReAct, Plan và Lai; không phụ thuộc quyết định LLM."""
from __future__ import annotations

import json
import time
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass

from langgraph.types import interrupt
from pydantic import ValidationError

from domain import BookingRequest, MockAirline, TOOL_SCHEMAS, violations


def as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


@dataclass
class Limits:
    model_calls: int = 30
    tool_calls: int = 30
    seconds: float = 60.0
    repeat_k: int = 3
    stall_n: int = 8
    poll_limit: int = 4
    replans: int = 8


class Harness:
    def __init__(self, request: BookingRequest, backend: MockAirline, limits: Limits | None = None):
        self.request, self.backend = request, backend
        self.limits = limits or Limits()
        self.started = time.perf_counter()
        self.trace: list[dict] = []
        self.observations: list[dict] = []
        self.status: str | None = None
        self.reason = ""
        self.verified_booking: dict | None = None
        self.model_calls = self.tool_calls = self.attempts = self.replans = 0
        self.input_tokens = self.output_tokens = 0
        self.usage_calls = 0
        self.recent = deque(maxlen=8)
        self.facts: set[str] = set()
        self.stall = 0
        self.polls = Counter()
        self.approved: set[str] = set()
        self.approval_requests: list[dict] = []
        self.approval_decisions: list[dict] = []
        self.pending_approval: dict | None = None
        self.quotes: dict[str, dict] = {}
        self.known_flights: set[str] = set()

    def log(self, kind: str, **data):
        self.trace.append(dict(seq=len(self.trace) + 1, kind=kind, **deepcopy(data)))

    def stop(self, status: str, reason: str):
        if self.status is None:
            self.status, self.reason = status, reason
            self.log("stop", status=status, reason=reason)

    def budget(self, *, model=False, tool=False) -> bool:
        if self.status:
            return False
        if time.perf_counter() - self.started >= self.limits.seconds:
            self.stop("BUDGET_EXCEEDED", "Hết thời gian chạy cho phép.")
        elif model and self.model_calls >= self.limits.model_calls:
            self.stop("BUDGET_EXCEEDED", "Hết ngân sách gọi model.")
        elif tool and self.tool_calls >= self.limits.tool_calls:
            self.stop("BUDGET_EXCEEDED", "Hết ngân sách gọi tool.")
        return self.status is None

    def before_model(self) -> bool:
        if not self.budget(model=True):
            return False
        self.model_calls += 1
        return True

    def record_model(self, message, phase: str):
        usage = getattr(message, "usage_metadata", None)
        if usage:
            self.usage_calls += 1
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
        self.log("model", phase=phase, content=message.content,
                 tool_calls=getattr(message, "tool_calls", []), usage=usage)

    @staticmethod
    def approval_key(record: dict) -> str:
        # Duyệt gắn với chính chuyến/giá/điều kiện, không phải một cờ True toàn phiên.
        keys = ("flight_id", "origin", "destination", "depart_date", "departure", "price", "refundable")
        return as_json({k: record.get(k) for k in keys})

    def requires_approval(self, record: dict) -> bool:
        return (record["price"] > self.request.auto_approval_limit
                or (self.request.approve_nonrefundable and not record["refundable"]))

    def authorize(self, name: str, record: dict) -> bool:
        if not self.requires_approval(record):
            return True
        key = self.approval_key(record)
        if key in self.approved:
            return True
        payload = dict(action=name, flight_id=record["flight_id"], price=record["price"],
                       refundable=record["refundable"], approval_key=key,
                       authorized_actions=["hold_seat", "pay"],
                       reason="Vượt hạn mức tự động hoặc vé không hoàn.",
                       question="Đồng ý giữ chỗ VÀ thanh toán giả lập với đúng chuyến, giá và điều kiện này?")
        # Node được chạy lại khi resume: chỉ ghi yêu cầu một lần, không ghi side effect trước interrupt.
        if not self.pending_approval or self.pending_approval["approval_key"] != key:
            self.pending_approval = payload
            self.approval_requests.append(deepcopy(payload))
            self.log("approval_request", **payload)
        answer = interrupt(payload)
        approved = isinstance(answer, dict) and answer.get("approved") is True and answer.get("approval_key") == key
        self.approval_decisions.append(dict(approval_key=key, approved=approved))
        self.log("approval_decision", approval_key=key, approved=approved)
        self.pending_approval = None
        if not approved:
            self.stop("APPROVAL_REJECTED", "Người duyệt từ chối hoặc phê duyệt không khớp hành động.")
            return False
        self.approved.add(key)
        return True

    def error(self, code: str, hint: str, retryable=False):
        return dict(status="error", error_code=code, hint=hint, retryable=retryable)

    def execute(self, name: str, args: dict) -> dict:
        """Cổng duy nhất để cả ba agent gọi backend. Không catch interrupt()."""
        if not self.budget(tool=True):
            return self.error(self.status or "STOPPED", self.reason)
        schema = TOOL_SCHEMAS.get(name)
        if schema is None:
            self.stop("INVALID_ACTION", f"Tool không tồn tại: {name}")
            return self.error("INVALID_TOOL", self.reason)
        try:
            args = schema.model_validate(args).model_dump(mode="json")
        except ValidationError as exc:
            self.stop("INVALID_ACTION", "Tham số tool không đúng schema.")
            self.log("invalid_args", tool=name, errors=exc.errors(include_url=False))
            return self.error("INVALID_ARGS", self.reason)

        record = None
        if name == "hold_seat":
            fid = args["flight_id"]
            if any(b["flight_id"] != fid for b in self.backend.bookings.values()):
                return self.record(name, args, self.error("ACTIVE_BOOKING_EXISTS", "Đã có booking; bàn giao trước khi đặt chuyến khác."), executed=False)
            record = self.backend.flights.get(fid)
            quote = self.quotes.get(fid)
            if fid not in self.known_flights or quote is None or record is None:
                return self.record(name, args, self.error("MISSING_QUOTE", "Tìm chuyến và kiểm ghế trước khi giữ chỗ."), executed=False)
            if violations(record, self.request):
                return self.record(name, args, self.error("CONSTRAINT_VIOLATION", "Chọn chuyến đáp ứng đủ ràng buộc."), executed=False)
            if args["quoted_price"] != quote["price"] or self.approval_key(quote) != self.approval_key(record):
                return self.record(name, args, self.error("STALE_QUOTE", "Giá hoặc điều kiện đã đổi; kiểm tra lại.", True), executed=False)
        elif name == "pay":
            record = self.backend.bookings.get(args["booking_id"])
            if record is None:
                return self.record(name, args, self.error("UNKNOWN_BOOKING", "Cần giữ chỗ trước khi thanh toán."), executed=False)
            if violations(record, self.request):
                return self.record(name, args, self.error("CONSTRAINT_VIOLATION", "Booking không đáp ứng yêu cầu."), executed=False)
        elif name == "search_flights":
            if (args["origin"], args["destination"], args["depart_date"]) != (
                self.request.origin, self.request.destination, self.request.depart_date.isoformat()
            ):
                return self.record(name, args, self.error("CONSTRAINT_VIOLATION", "Không tự đổi tuyến hoặc ngày yêu cầu."), executed=False)
        if record is not None and not self.authorize(name, record):
            return self.error("APPROVAL_REJECTED", self.reason)
        # Kiểm lại thời gian sau thời gian chờ duyệt, trước khi có tác dụng phụ.
        if not self.budget(tool=True):
            return self.error(self.status, self.reason)
        result = self.backend.execute(name, args)
        self.tool_calls += 1
        return self.record(name, args, result, executed=True)

    def record(self, name: str, args: dict, result: dict, *, executed: bool) -> dict:
        self.attempts += 1
        entry = dict(tool=name, args=deepcopy(args), result=deepcopy(result), executed=executed)
        self.observations.append(entry)
        self.log("tool", **entry)
        before = len(self.facts)
        if result.get("status") == "ok":
            if name == "search_flights":
                self.known_flights.update(f["flight_id"] for f in result["flights"])
                self.facts.add("searched")
            if name == "check_seat":
                f = result["flight"]
                self.quotes[f["flight_id"]] = deepcopy(f)
                self.facts.add("checked:" + f["flight_id"])
            if "booking" in result:
                b = result["booking"]
                self.facts.add("held:" + b["booking_id"])
                if b["paid"]:
                    self.facts.add("paid:" + b["booking_id"])
                # Chỉ observation đọc lại mới có thể kết luận thành công.
                if name == "get_booking" and self.is_complete(b):
                    self.verified_booking = deepcopy(b)
                    self.stop("SUCCESS", "Đã đọc lại booking: confirmed, paid và đúng mọi ràng buộc.")
        if self.status:
            return result
        progressed = len(self.facts) > before
        fp = as_json([name, args])
        polling = (name == "get_booking" and result.get("status") == "ok"
                   and result.get("booking", {}).get("status") == "pending_confirmation")
        if polling:
            self.polls[args["booking_id"]] += 1
            if self.polls[args["booking_id"]] >= self.limits.poll_limit:
                self.stop("STALL", "Đã trả tiền nhưng trạng thái chưa xác nhận trong giới hạn polling.")
        elif not progressed and self.recent.count(fp) + 1 >= self.limits.repeat_k:
            self.stop("LOOP_DETECTED", f"Cùng hành động xuất hiện {self.limits.repeat_k} lần, không có tiến triển: {fp}")
        self.recent.append(fp)
        self.stall = 0 if progressed or polling else self.stall + 1
        if self.stall >= self.limits.stall_n:
            self.stop("STALL", "Không có thêm thông tin hữu ích hoặc bước đặt vé mới.")
        return result

    def is_complete(self, booking: dict) -> bool:
        actual = self.backend.bookings.get(booking.get("booking_id"))
        return (actual is not None and actual == booking and not violations(booking, self.request)
                and booking.get("status") == "confirmed" and booking.get("paid") is True
                and self.backend.charges.get(booking["booking_id"]) == booking["price"])

    def no_match_proven(self) -> bool:
        """Bằng chứng trong thế giới mock hữu hạn, không suy ra từ lời nói của model."""
        searches = [o for o in self.observations if o["tool"] == "search_flights"
                    and o["result"].get("status") == "ok" and o["result"].get("complete")]
        if not searches:
            return False
        for f in searches[-1]["result"]["flights"]:
            current = self.quotes.get(f["flight_id"], f)
            sold_out = any(o["tool"] == "hold_seat" and o["args"].get("flight_id") == f["flight_id"]
                           and o["result"].get("error_code") == "SOLD_OUT" for o in self.observations)
            if sold_out:
                continue
            if current["seats"] > 0 and not violations(current, self.request):
                return False
        return True

    def finish(self):
        if self.status:
            return
        if self.no_match_proven():
            self.stop("NO_SUITABLE_FLIGHT", "Đã duyệt danh sách đầy đủ trong dữ liệu mock; không có chuyến thỏa yêu cầu.")
        else:
            self.stop("UNVERIFIED_COMPLETION", "Model dừng nhưng chưa có booking được đọc lại và kiểm chứng.")

    def handoff(self) -> dict:
        questions = {
            "NEEDS_APPROVAL": "Bạn có duyệt đúng hành động, chuyến và giá trong pending_approval không?",
            "NO_SUITABLE_FLIGHT": "Bạn muốn đổi ngày/khung giờ hoặc trần giá nào?",
            "APPROVAL_REJECTED": "Bạn muốn chọn điều kiện vé khác hay kết thúc yêu cầu?",
            "PLAN_FAILED": "Kế hoạch đã hỏng ở bước ghi trong trace; bạn muốn thử chuyến khác hoặc lập lại kế hoạch?",
            "LOOP_DETECTED": "Nguồn dữ liệu đang lặp lỗi; bạn muốn đổi nguồn hay thử lại sau?",
        }
        return dict(stop_reason=self.status, explanation=self.reason,
                    request=self.request.model_dump(mode="json"),
                    tried=[dict(tool=o["tool"], args=o["args"], status=o["result"].get("status"),
                                error_code=o["result"].get("error_code")) for o in self.observations],
                    bookings=deepcopy(self.backend.bookings), charges=deepcopy(self.backend.charges),
                    side_effects=deepcopy(self.backend.audit), pending_approval=self.pending_approval,
                    question=None if self.status == "SUCCESS" else questions.get(
                        self.status, "Bạn muốn tiếp tục từ trạng thái đã ghi, đổi nguồn dữ liệu hay dừng tác vụ?"))

    def result(self, strategy: str, scenario: str, model_name: str) -> dict:
        audit_violations = sum(bool(violations(e["booking"], self.request)) for e in self.backend.audit)
        unauthorized = sum(self.requires_approval(e["booking"]) and self.approval_key(e["booking"]) not in self.approved
                           for e in self.backend.audit)
        return dict(strategy=strategy, scenario=scenario, model=model_name, status=self.status,
                    verified_booking=self.verified_booking,
                    metrics=dict(model_calls=self.model_calls, tool_calls=self.tool_calls,
                                 tool_attempts=self.attempts, replans=self.replans,
                                 elapsed_ms=round((time.perf_counter() - self.started) * 1000, 3),
                                 input_tokens=self.input_tokens if self.usage_calls else None,
                                 output_tokens=self.output_tokens if self.usage_calls else None,
                                 usage_reported_calls=self.usage_calls,
                                 cost_usd=None, constraint_violations=audit_violations,
                                 unauthorized_actions=unauthorized,
                                 approval_requests=len(self.approval_requests),
                                 charges=len(self.backend.charges)),
                    handoff=self.handoff(), trace=self.trace)
