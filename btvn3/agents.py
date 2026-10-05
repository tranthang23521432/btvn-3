"""Ba kiến trúc thật chạy trên LangChain/LangGraph; cùng một model và harness.

MockDecisionModel chỉ nhìn request và observation. Nó không nhận tên kịch bản,
dữ liệu backend chưa được đọc, hay kết quả kỳ vọng của bộ chấm.
"""
from __future__ import annotations

import json
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from domain import BookingRequest, MockAirline, Scenario, TOOL_DESCRIPTIONS, TOOL_SCHEMAS, violations
from harness import Harness, Limits, as_json


ToolName = Literal["search_flights", "check_seat", "hold_seat", "pay", "get_booking"]


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: ToolName
    args: dict[str, Any]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[Action] = Field(max_length=12)


def visible_state(observations: list[dict], request: BookingRequest) -> dict:
    """Tóm trạng thái từ dữ liệu mà model thực sự đã nhận."""
    flights, quotes, blocked = None, {}, set()
    booking = None
    for o in observations:
        name, args, r = o["tool"], o["args"], o["result"]
        if name == "search_flights" and r.get("status") == "ok":
            flights = r["flights"]
        if name == "check_seat" and r.get("status") == "ok":
            quotes[args["flight_id"]] = r["flight"]
        if name == "hold_seat" and r.get("error_code") == "SOLD_OUT":
            blocked.add(args["flight_id"])
        if name == "hold_seat" and r.get("error_code") in ("PRICE_CHANGED", "STALE_QUOTE"):
            quotes.pop(args["flight_id"], None)
        if "booking" in r:
            booking = r["booking"]
    candidates = [quotes.get(f["flight_id"], f) for f in (flights or [])]
    candidates = sorted((f for f in candidates if f["flight_id"] not in blocked
                         and f["seats"] > 0 and not violations(f, request)), key=lambda f: (f["price"], f["flight_id"]))
    return dict(searched=flights is not None, candidates=candidates, quotes=quotes, booking=booking)


def next_action(obs: list[dict], req: BookingRequest) -> dict | None:
    """Chính sách minh họa cố định của model giả; không phải suy luận của LLM."""
    s = visible_state(obs, req)
    if s["booking"]:
        b = s["booking"]
        if b["status"] == "confirmed" and obs[-1]["tool"] == "get_booking":
            return None
        return dict(tool="get_booking" if b["paid"] else "pay", args=dict(booking_id=b["booking_id"]))
    if not s["searched"]:
        return dict(tool="search_flights", args=dict(origin=req.origin, destination=req.destination,
                                                     depart_date=req.depart_date.isoformat()))
    if not s["candidates"]:
        return None
    f = s["candidates"][0]
    if f["flight_id"] not in s["quotes"]:
        return dict(tool="check_seat", args=dict(flight_id=f["flight_id"]))
    return dict(tool="hold_seat", args=dict(flight_id=f["flight_id"], quoted_price=f["price"]))


def mock_plan(obs: list[dict], req: BookingRequest) -> dict:
    action = next_action(obs, req)
    if action is None:
        return {"steps": []}
    chain = [dict(tool="search_flights", args=dict(origin=req.origin, destination=req.destination,
                                                 depart_date=req.depart_date.isoformat())),
             dict(tool="check_seat", args={"flight_id": "$candidate"}),
             dict(tool="hold_seat", args={"flight_id": "$candidate", "quoted_price": "$quote_price"}),
             dict(tool="pay", args={"booking_id": "$booking"}),
             dict(tool="get_booking", args={"booking_id": "$booking"})]
    start = [a["tool"] for a in chain].index(action["tool"])
    suffix = chain[start:]
    suffix[0] = action
    return {"steps": suffix}


class MockDecisionModel(BaseChatModel):
    """Một model giả duy nhất cho cả ba mẫu; không token thật và không phí API."""
    @property
    def _llm_type(self) -> str:
        return "se373-booking-mock"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        humans = [m for m in messages if m.type == "human"]
        payload = json.loads(humans[-1].content)
        req = BookingRequest.model_validate(payload["request"])
        planning = any("MODE=PLAN" in str(m.content) for m in messages if m.type == "system")
        if planning:
            msg = AIMessage(content=as_json(mock_plan(payload["observations"], req)))
        else:
            call_map = {c["id"]: c for m in messages for c in (getattr(m, "tool_calls", None) or [])}
            obs = []
            for m in messages:
                if m.type == "tool" and m.tool_call_id in call_map:
                    c = call_map[m.tool_call_id]
                    obs.append(dict(tool=c["name"], args=c["args"], result=json.loads(m.content)))
            action = next_action(obs, req)
            msg = AIMessage(content="Đã kết thúc lựa chọn; harness cần kiểm chứng kết quả.") if action is None else AIMessage(
                content="", tool_calls=[dict(name=action["tool"], args=action["args"], id=f"call_{len(obs)+1}", type="tool_call")])
        return ChatResult(generations=[ChatGeneration(message=msg)])


COMMON_PROMPT = """Bạn là agent đặt vé trong môi trường giả lập cho một hành khách.
request là ràng buộc bắt buộc. Không tự đổi ngày, tuyến, giờ, trần giá.
Chỉ dùng flight_id, booking_id và giá đã có từ tool. Kiểm ghế trước khi giữ.
Không xem thông báo trả tiền là hoàn tất: phải get_booking để xác nhận.
Lỗi retryable có thể thử lại có giới hạn; lỗi hết ghế/không đạt yêu cầu cần chọn chuyến khác.
Không gọi tool song song. Harness sẽ kiểm quyền, ngân sách và phê duyệt.
"""


class SharedMiddleware(AgentMiddleware):
    def __init__(self, harness: Harness):
        self.h = harness

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if not self.h.before_model():
            return {"jump_to": "end"}
        return None

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        msg = state["messages"][-1]
        self.h.record_model(msg, "react")
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            self.h.finish()
            return {"jump_to": "end"}
        # Serialize business writes; model cannot race pay and hold_seat.
        if len(calls) != 1:
            self.h.stop("INVALID_ACTION", "Chỉ cho một tool call mỗi lượt để giữ thứ tự nghiệp vụ.")
        else:
            c = calls[0]
            schema = TOOL_SCHEMAS.get(c["name"])
            try:
                if schema is None:
                    raise ValueError("unknown tool")
                schema.model_validate(c["args"])
            except (ValueError, ValidationError):
                self.h.stop("INVALID_ACTION", "Tên tool hoặc tham số model đề xuất không hợp lệ.")
        return {"jump_to": "end"} if self.h.status else None


def build_tools(h: Harness):
    def make(name):
        def execute(**kwargs):
            return h.execute(name, kwargs)
        return StructuredTool.from_function(func=execute, name=name,
                                            description=TOOL_DESCRIPTIONS[name], args_schema=TOOL_SCHEMAS[name])
    return [make(name) for name in TOOL_SCHEMAS]


class PlanState(TypedDict, total=False):
    plan: list[dict]
    index: int
    chunk: int
    selected: str | None
    event: str


def build_planned_agent(model, h: Harness, hybrid: bool, chunk_size=2):
    tool_map = {t.name: t for t in build_tools(h)}

    def planner(state):
        if not h.before_model():
            return {"event": "stop"}
        if state.get("plan") is not None:
            if h.replans >= h.limits.replans:
                h.stop("BUDGET_EXCEEDED", "Đã chạm giới hạn lập lại kế hoạch.")
                # Chưa thực sự gọi model ở nhánh này.
                h.model_calls -= 1
                return {"event": "stop"}
            h.replans += 1
        prompt = COMMON_PROMPT + """
MODE=PLAN. Trả duy nhất JSON {"steps":[{"tool":"...","args":{...}}]}.
Lập kế hoạch các bước còn lại; được trả steps rỗng nếu không có đường thực hiện.
Tool schemas: """ + as_json({k: v.model_json_schema() for k, v in TOOL_SCHEMAS.items()}) + """
Khi chưa có kết quả tool, dùng placeholder trong args:
"$candidate": mã chuyến hợp lệ rẻ nhất sau tìm kiếm; được giữ cố định trong kế hoạch.
"$quote_price": giá check_seat của chuyến đã chọn; "$booking": mã booking vừa giữ.
Sau check_seat có thể hold_seat, pay, get_booking. Không tự bịa ID hay giá.
"""
        msg = model.invoke([SystemMessage(content=prompt), HumanMessage(content=as_json(
            dict(request=h.request.model_dump(mode="json"), observations=h.observations)))])
        h.record_model(msg, "replan" if h.replans else "plan")
        try:
            plan = Plan.model_validate_json(msg.content).model_dump()["steps"]
        except (ValueError, TypeError):
            h.stop("INVALID_PLAN", "Model không trả JSON kế hoạch đúng schema.")
            return {"event": "stop"}
        h.log("plan", steps=plan, replan=h.replans)
        if not plan:
            h.finish()
        # Concrete check/hold in first step fixes candidate for subsequent placeholders.
        first = plan[0] if plan else {}
        selected = first.get("args", {}).get("flight_id")
        if selected == "$candidate":
            selected = None
        return dict(plan=plan, index=0, chunk=0, selected=selected, event="continue")

    def executor(state):
        action = state["plan"][state["index"]]
        s = visible_state(h.observations, h.request)
        selected = state.get("selected")
        if selected is None and s["candidates"]:
            selected = s["candidates"][0]["flight_id"]
        args = dict(action["args"])
        placeholders = {"$candidate": selected,
                        "$quote_price": s["quotes"].get(selected, {}).get("price"),
                        "$booking": (s["booking"] or {}).get("booking_id")}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                if value not in placeholders or placeholders[value] is None:
                    if h.no_match_proven():
                        h.finish()
                    else:
                        h.stop("PLAN_FAILED", f"Không giải được placeholder {value} ở bước {state['index'] + 1}.")
                    return {"event": "stop"}
                args[key] = placeholders[value]
        # Hàm validate trong harness cũng được dùng cho ReAct. Tool.invoke dùng wrapper LangChain.
        try:
            TOOL_SCHEMAS[action["tool"]].model_validate(args)
        except ValidationError:
            h.stop("INVALID_ACTION", "Tham số của bước kế hoạch không hợp lệ sau khi gắn dữ liệu.")
            return {"event": "stop"}
        result = tool_map[action["tool"]].invoke(args)
        failed = result.get("status") != "ok"
        if action["tool"] == "check_seat" and result.get("status") == "ok":
            f = result["flight"]
            failed = f["seats"] <= 0 or bool(violations(f, h.request))
        if failed and not h.status:
            h.log("plan_step_failed", index=state["index"], action=action, observation=result)
            if not hybrid:
                h.stop("PLAN_FAILED", "Kế hoạch cố định gặp bước thất bại; bàn giao thay vì tự đổi kế hoạch.")
            return dict(event="replan", chunk=0, selected=selected)
        # Polling là một phần của bước xác minh, có giới hạn chung cho cả ba mẫu.
        pending = (action["tool"] == "get_booking" and result.get("booking", {}).get("status") == "pending_confirmation")
        return dict(index=state["index"] + (0 if pending else 1), chunk=state["chunk"] + 1,
                    selected=selected, event="poll" if pending else "continue")

    def after_plan(state):
        return END if h.status or not state.get("plan") else "execute"

    def after_execute(state):
        if h.status:
            return END
        if state["event"] == "poll":
            return "execute"
        if hybrid and (state["event"] == "replan" or state["chunk"] >= chunk_size):
            return "plan"
        if state["index"] >= len(state["plan"]):
            h.finish()
            return END
        return "execute"

    graph = StateGraph(PlanState)
    graph.add_node("plan", planner)
    graph.add_node("execute", executor)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan)
    graph.add_conditional_edges("execute", after_execute)
    return graph.compile(checkpointer=InMemorySaver())


def make_model(name: str):
    if name == "mock":
        return MockDecisionModel()
    from langchain.chat_models import init_chat_model
    return init_chat_model(name, temperature=0, timeout=30, max_retries=0)


def run_agent(strategy: str, scenario: Scenario, *, model_name="mock", model=None,
              request: BookingRequest | None = None, limits: Limits | None = None,
              approval: str | None = None, chunk_size=2) -> dict:
    if strategy not in ("react", "plan", "hybrid"):
        raise ValueError("strategy phải là react, plan hoặc hybrid")
    h = Harness(request or BookingRequest(), MockAirline(scenario), limits)
    model = model if model is not None else make_model(model_name)
    if strategy == "react":
        app = create_agent(model=model, tools=build_tools(h), system_prompt=COMMON_PROMPT,
                           middleware=[SharedMiddleware(h)], checkpointer=InMemorySaver())
        initial = {"messages": [HumanMessage(content=as_json(dict(request=h.request.model_dump(mode="json"))))]}
    else:
        app = build_planned_agent(model, h, strategy == "hybrid", chunk_size)
        initial = {}
    config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 300}
    # Chế độ đánh giá cấp quyết định mock từ bên ngoài model, giống nhau cho cả ba mẫu.
    decision_mode = approval or scenario.approval
    try:
        outcome = app.invoke(initial, config)
        while outcome.get("__interrupt__"):
            payload = outcome["__interrupt__"][0].value
            decision = decision_mode
            if decision == "ask":
                print("\nCẦN PHÊ DUYỆT (chỉ giả lập):", as_json(payload))
                reply = input("Nhập yes để đồng ý, no để từ chối, Enter để chờ: ").strip().lower()
                decision = {"yes": "approve", "no": "reject"}.get(reply, "defer")
            if decision == "defer":
                h.stop("NEEDS_APPROVAL", "Chưa thực thi hành động đang chờ phê duyệt.")
                break
            outcome = app.invoke(Command(resume={"approved": decision == "approve",
                                                 "approval_key": payload["approval_key"]}), config)
        h.finish()
    except GraphRecursionError:
        h.stop("BUDGET_EXCEEDED", "Chạm chặn cứng số bước graph; xem trace để tìm nguyên nhân.")
    except Exception as exc:
        h.log("exception", error_type=type(exc).__name__, message=str(exc))
        h.stop("EXECUTION_ERROR", f"{type(exc).__name__}: {exc}")
    return h.result(strategy, scenario.name, model_name)
