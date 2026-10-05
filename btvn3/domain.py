"""Dữ liệu nghiệp vụ và backend vé giả lập. Không gọi mạng, không thu tiền."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BookingRequest(BaseModel):
    """Ràng buộc bất biến trong một phiên; giá là tổng tiền VND cho một khách."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    origin: str = "SGN"
    destination: str = "DAD"
    depart_date: date = date(2026, 10, 7)
    depart_before: time = time(12, 0)
    max_price: int = Field(default=2_000_000, gt=0)
    auto_approval_limit: int = Field(default=1_500_000, gt=0)
    approve_nonrefundable: bool = True


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str
    destination: str
    depart_date: date


class FlightArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flight_id: str


class HoldArgs(FlightArgs):
    quoted_price: int = Field(gt=0)


class BookingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    booking_id: str


TOOL_SCHEMAS = {
    "search_flights": SearchArgs, "check_seat": FlightArgs,
    "hold_seat": HoldArgs, "pay": BookingArgs, "get_booking": BookingArgs,
}
TOOL_DESCRIPTIONS = {
    "search_flights": "Tìm toàn bộ chuyến bay theo tuyến và ngày; giá và ghế có thể đổi.",
    "check_seat": "Đọc giá, ghế và điều kiện vé hiện tại trước khi giữ chỗ.",
    "hold_seat": "Giữ một ghế ở giá vừa kiểm tra; cần flight_id và quoted_price.",
    "pay": "Thanh toán GIẢ LẬP một booking đang giữ. Có thể cần người phê duyệt.",
    "get_booking": "Đọc lại booking để xác nhận confirmed, paid và các ràng buộc.",
}


def violations(flight: dict, req: BookingRequest) -> list[str]:
    """Không dùng LLM để kiểm ràng buộc; dữ liệu thiếu cũng là không đạt."""
    errors = []
    for key, expected in (("origin", req.origin), ("destination", req.destination),
                          ("depart_date", req.depart_date.isoformat())):
        if flight.get(key) != expected:
            errors.append(key)
    try:
        if time.fromisoformat(flight["departure"]) >= req.depart_before:
            errors.append("departure")
    except (KeyError, TypeError, ValueError):
        errors.append("departure")
    price = flight.get("price")
    if type(price) is not int or not 0 < price <= req.max_price:
        errors.append("price")
    return errors


def flight(fid="VN101", price=1_200_000, **changes) -> dict:
    return dict(flight_id=fid, origin="SGN", destination="DAD",
                depart_date="2026-10-07", departure="08:00:00", price=price,
                seats=2, refundable=True) | changes


@dataclass
class Scenario:
    name: str
    description: str
    flights: list[dict] = field(default_factory=lambda: [flight(), flight("VJ202", 1_400_000)])
    sold_out_on_check: set[str] = field(default_factory=set)
    price_on_check: dict[str, int] = field(default_factory=dict)
    sold_out_on_hold: set[str] = field(default_factory=set)
    check_timeouts: int = 0
    persistent_timeout: bool = False
    confirmation_reads: int = 0
    approval: Literal["approve", "reject", "defer"] = "approve"
    expected: dict[str, str] = field(default_factory=dict)
    feasible: bool = True


def scenarios() -> dict[str, Scenario]:
    """Expected dùng riêng cho bộ chấm; agent không được đọc tên/kịch bản này."""
    def expect(react, plan=None, hybrid=None):
        return dict(react=react, plan=plan or react, hybrid=hybrid or react)
    rows = [
        Scenario("normal", "Có vé đúng yêu cầu", expected=expect("SUCCESS")),
        Scenario("sold_out", "Chuyến rẻ nhất hết ghế khi kiểm tra",
                 sold_out_on_check={"VN101"}, expected=expect("SUCCESS", "PLAN_FAILED")),
        Scenario("price_jump", "Giá chuyến đầu tăng vượt ngân sách khi kiểm tra",
                 price_on_check={"VN101": 2_300_000}, expected=expect("SUCCESS", "PLAN_FAILED")),
        Scenario("no_match", "Tất cả vé vượt trần giá",
                 flights=[flight(price=2_300_000), flight("VJ202", 2_500_000)],
                 expected=expect("NO_SUITABLE_FLIGHT"), feasible=False),
        Scenario("approval_yes", "Vé không hoàn, người duyệt đồng ý",
                 flights=[flight(price=1_800_000, refundable=False)], expected=expect("SUCCESS")),
        Scenario("approval_no", "Người duyệt từ chối trước khi giữ vé không hoàn",
                 flights=[flight(price=1_800_000, refundable=False)], approval="reject",
                 expected=expect("APPROVAL_REJECTED"), feasible=False),
        Scenario("approval_wait", "Chờ quyết định của con người",
                 flights=[flight(price=1_800_000, refundable=False)], approval="defer",
                 expected=expect("NEEDS_APPROVAL"), feasible=False),
        Scenario("transient_error", "check_seat timeout một lần rồi hồi phục",
                 check_timeouts=1, expected=expect("SUCCESS", "PLAN_FAILED")),
        Scenario("persistent_error", "check_seat timeout liên tục",
                 persistent_timeout=True, expected=expect("LOOP_DETECTED", "PLAN_FAILED"), feasible=False),
        Scenario("wrong_constraints", "Có vé rẻ sai giờ/ngày/tuyến và một vé phù hợp",
                 flights=[flight("LATE", 700_000, departure="15:00:00"),
                          flight("TOMORROW", 600_000, depart_date="2026-10-08"),
                          flight("OTHER", 500_000, destination="HAN"), flight("VALID", 1_400_000)],
                 expected=expect("SUCCESS")),
        Scenario("confirmation_delay", "Đã thanh toán, cần polling để được confirmed",
                 confirmation_reads=2, expected=expect("SUCCESS")),
        Scenario("hold_race", "Ghế bị lấy mất giữa check_seat và hold_seat",
                 sold_out_on_hold={"VN101"}, expected=expect("SUCCESS", "PLAN_FAILED")),
    ]
    return {s.name: s for s in rows}


class MockAirline:
    """Mỗi run có kho vé riêng. Các thao tác ghi có tính idempotent."""
    def __init__(self, scenario: Scenario):
        self.scenario = deepcopy(scenario)
        self.flights = {f["flight_id"]: deepcopy(f) for f in scenario.flights}
        self.bookings: dict[str, dict] = {}
        self.charges: dict[str, int] = {}
        self.check_calls = 0
        self.reads: dict[str, int] = {}
        self.audit: list[dict] = []

    @staticmethod
    def error(code, hint, retryable=False):
        return dict(status="error", error_code=code, hint=hint, retryable=retryable)

    def execute(self, name: str, args: dict) -> dict:
        if name == "search_flights":
            rows = [deepcopy(f) for f in self.flights.values()
                    if all(f[k] == args[k] for k in ("origin", "destination", "depart_date"))]
            return dict(status="ok", flights=rows, complete=True)
        if name in ("check_seat", "hold_seat"):
            f = self.flights.get(args["flight_id"])
            if f is None:
                return self.error("UNKNOWN_FLIGHT", "Dùng mã chuyến từ search_flights.")
            if name == "check_seat":
                self.check_calls += 1
                if self.scenario.persistent_timeout or self.check_calls <= self.scenario.check_timeouts:
                    return self.error("TIMEOUT", "Có thể thử lại với số lần giới hạn.", True)
                if f["flight_id"] in self.scenario.sold_out_on_check:
                    f["seats"] = 0
                if f["flight_id"] in self.scenario.price_on_check:
                    f["price"] = self.scenario.price_on_check[f["flight_id"]]
                return dict(status="ok", flight=deepcopy(f))
            # Giữ lại cùng một ghế khi một thao tác bị phát lại.
            prior = next((b for b in self.bookings.values() if b["flight_id"] == f["flight_id"]), None)
            if prior:
                return dict(status="ok", booking=deepcopy(prior), idempotent=True)
            if f["flight_id"] in self.scenario.sold_out_on_hold:
                f["seats"] = 0
            if f["seats"] <= 0:
                return self.error("SOLD_OUT", "Chọn chuyến khác.")
            if f["price"] != args["quoted_price"]:
                return self.error("PRICE_CHANGED", "Đọc lại giá bằng check_seat.", True)
            bid = f"BK{len(self.bookings) + 1:03d}"
            booking = deepcopy(f) | dict(booking_id=bid, status="held", paid=False)
            self.bookings[bid] = booking
            f["seats"] -= 1
            self.audit.append(dict(action="hold_seat", booking=deepcopy(booking)))
            return dict(status="ok", booking=deepcopy(booking))
        bid = args["booking_id"]
        b = self.bookings.get(bid)
        if b is None:
            return self.error("UNKNOWN_BOOKING", "Dùng booking_id do hold_seat trả về.")
        if name == "pay":
            if b["paid"]:
                return dict(status="ok", booking=deepcopy(b), idempotent=True)
            b["paid"] = True
            b["status"] = "pending_confirmation" if self.scenario.confirmation_reads else "confirmed"
            self.charges[bid] = b["price"]
            self.audit.append(dict(action="pay", booking=deepcopy(b)))
        elif name == "get_booking" and b["paid"]:
            self.reads[bid] = self.reads.get(bid, 0) + 1
            if self.reads[bid] > self.scenario.confirmation_reads:
                b["status"] = "confirmed"
        return dict(status="ok", booking=deepcopy(b))
