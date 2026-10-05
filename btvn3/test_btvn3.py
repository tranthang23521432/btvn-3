"""Kiểm thử nghiệp vụ, cổng quyền và ba graph; chạy bằng unittest (không cần pytest)."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command

from agents import MockDecisionModel, run_agent
from domain import BookingRequest, MockAirline, flight, scenarios, violations
from evaluate import judge
from harness import Harness, Limits


STRATEGIES = ("react", "plan", "hybrid")


class ClaimSuccessModel(MockDecisionModel):
    def _generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Đã đặt vé thành công, mã BK999."))])


class InvalidToolModel(MockDecisionModel):
    def _generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[
            dict(name="delete_database", args={}, id="bad", type="tool_call")]))])


class AgentTests(unittest.TestCase):
    def test_all_scenarios_all_architectures(self):
        for case in scenarios().values():
            for strategy in STRATEGIES:
                with self.subTest(case=case.name, strategy=strategy):
                    r = run_agent(strategy, case)
                    self.assertEqual(r["status"], case.expected[strategy], r["handoff"])
                    j = judge(r, case.feasible)
                    self.assertEqual(j["constraint_violations"], 0)
                    self.assertEqual(j["unauthorized_actions"], 0)
                    self.assertFalse(j["false_success"])
                    self.assertTrue(j["complete_handoff"])

    def test_model_cannot_claim_success_without_readback(self):
        r = run_agent("react", scenarios()["normal"], model=ClaimSuccessModel())
        self.assertEqual(r["status"], "UNVERIFIED_COMPLETION")
        self.assertFalse(r["handoff"]["charges"])

    def test_unknown_tool_stopped_before_execution(self):
        r = run_agent("react", scenarios()["normal"], model=InvalidToolModel())
        self.assertEqual(r["status"], "INVALID_ACTION")
        self.assertEqual(r["metrics"]["tool_calls"], 0)

    def test_plan_requires_valid_schema(self):
        for strategy in ("plan", "hybrid"):
            r = run_agent(strategy, scenarios()["normal"], model=ClaimSuccessModel())
            self.assertEqual(r["status"], "INVALID_PLAN")
            self.assertEqual(r["metrics"]["tool_calls"], 0)

    def test_approval_reject_and_defer_have_no_side_effect(self):
        for mode, expected in (("reject", "APPROVAL_REJECTED"), ("defer", "NEEDS_APPROVAL")):
            for strategy in STRATEGIES:
                with self.subTest(mode=mode, strategy=strategy):
                    r = run_agent(strategy, scenarios()["approval_yes"], approval=mode)
                    self.assertEqual(r["status"], expected)
                    self.assertEqual(r["handoff"]["side_effects"], [])
                    self.assertEqual(r["handoff"]["bookings"], {})

    def test_interrupt_resume_does_not_duplicate_charge(self):
        for strategy in STRATEGIES:
            r = run_agent(strategy, scenarios()["approval_yes"], approval="approve")
            self.assertEqual(r["status"], "SUCCESS")
            self.assertEqual(r["metrics"]["approval_requests"], 1)
            self.assertEqual(r["metrics"]["charges"], 1)
            self.assertEqual(len(r["handoff"]["side_effects"]), 2)
            request = next(e for e in r["trace"] if e["kind"] == "approval_request")
            self.assertEqual(request["authorized_actions"], ["hold_seat", "pay"])
            self.assertEqual(judge(r, True)["unauthorized_actions"], 0)

    def test_manual_no_rejects_and_yes_resumes(self):
        for reply, expected in (("no", "APPROVAL_REJECTED"), ("yes", "SUCCESS"), ("", "NEEDS_APPROVAL")):
            with patch("builtins.input", return_value=reply), patch("builtins.print"):
                r = run_agent("hybrid", scenarios()["approval_yes"], approval="ask")
            self.assertEqual(r["status"], expected)

    def test_budget_blocks_before_next_tool(self):
        for strategy in STRATEGIES:
            r = run_agent(strategy, scenarios()["normal"], limits=Limits(tool_calls=2))
            self.assertEqual(r["status"], "BUDGET_EXCEEDED")
            self.assertEqual(r["metrics"]["tool_calls"], 2)
            self.assertEqual(r["handoff"]["side_effects"], [])

    def test_model_budget(self):
        r = run_agent("react", scenarios()["normal"], limits=Limits(model_calls=1))
        self.assertEqual(r["status"], "BUDGET_EXCEEDED")
        self.assertEqual(r["metrics"]["model_calls"], 1)

    def test_polling_is_bounded_and_paid_state_is_handed_off(self):
        for strategy in STRATEGIES:
            case = deepcopy(scenarios()["confirmation_delay"])
            case.confirmation_reads = 100
            r = run_agent(strategy, case, limits=Limits(poll_limit=2))
            self.assertEqual(r["status"], "STALL")
            self.assertEqual(r["metrics"]["charges"], 1)
            self.assertTrue(r["handoff"]["bookings"]["BK001"]["paid"])
            self.assertEqual(r["handoff"]["bookings"]["BK001"]["status"], "pending_confirmation")

    def test_plan_is_once_and_hybrid_replans(self):
        for strategy, expected in (("react", 5), ("plan", 1), ("hybrid", 3)):
            r = run_agent(strategy, scenarios()["normal"])
            self.assertEqual(r["metrics"]["model_calls"], expected)
            self.assertEqual(r["verified_booking"]["flight_id"], "VN101")


class HarnessTests(unittest.TestCase):
    def make(self, case="normal", **limits):
        return Harness(BookingRequest(), MockAirline(scenarios()[case]), Limits(**limits))

    def search(self, h):
        return h.execute("search_flights", dict(origin="SGN", destination="DAD", depart_date="2026-10-07"))

    def prepare(self, h, fid="VN101"):
        self.search(h)
        return h.execute("check_seat", dict(flight_id=fid))

    def test_hard_constraints_cannot_be_overridden_by_approval(self):
        h = self.make("no_match")
        self.prepare(h)
        h.approved.add(h.approval_key(h.backend.flights["VN101"]))
        result = h.execute("hold_seat", dict(flight_id="VN101", quoted_price=2_300_000))
        self.assertEqual(result["error_code"], "CONSTRAINT_VIOLATION")
        self.assertEqual(h.backend.audit, [])

    def test_wrong_date_and_time_rejected(self):
        req = BookingRequest()
        for f, field in ((flight(departure="12:00:00"), "departure"),
                         (flight(depart_date="2026-10-08"), "depart_date"),
                         (flight(destination="HAN"), "destination")):
            self.assertIn(field, violations(f, req))
        h = self.make()
        r = h.execute("search_flights", dict(origin="SGN", destination="DAD", depart_date="2026-10-08"))
        self.assertEqual(r["error_code"], "CONSTRAINT_VIOLATION")
        self.assertEqual(h.tool_calls, 0)

    def test_missing_quote_cannot_book(self):
        h = self.make()
        r = h.execute("hold_seat", dict(flight_id="VN101", quoted_price=1_200_000))
        self.assertEqual(r["error_code"], "MISSING_QUOTE")
        self.assertFalse(h.backend.bookings)

    def test_stale_quote_cannot_book(self):
        h = self.make()
        self.prepare(h)
        h.backend.flights["VN101"]["price"] += 100_000
        r = h.execute("hold_seat", dict(flight_id="VN101", quoted_price=1_200_000))
        self.assertEqual(r["error_code"], "STALE_QUOTE")
        self.assertFalse(h.backend.bookings)

    def test_price_change_requires_new_approval(self):
        h = self.make("approval_yes")
        self.prepare(h)
        old_key = h.approval_key(h.backend.flights["VN101"])
        h.approved.add(old_key)
        h.backend.flights["VN101"]["price"] = 1_900_000
        h.execute("check_seat", dict(flight_id="VN101"))
        graph = StateGraph(dict)
        graph.add_node("hold", lambda state: h.execute("hold_seat", dict(flight_id="VN101", quoted_price=1_900_000)))
        graph.add_edge(START, "hold")
        graph.add_edge("hold", END)
        app = graph.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "approval-scope"}}
        pending = app.invoke({}, config)["__interrupt__"][0].value
        self.assertNotEqual(pending["approval_key"], old_key)
        self.assertFalse(h.backend.audit)
        app.invoke(Command(resume=dict(approved=True, approval_key=old_key)), config)
        self.assertEqual(h.status, "APPROVAL_REJECTED")
        self.assertFalse(h.backend.audit)

    def test_no_second_booking_for_same_request(self):
        h = self.make()
        self.prepare(h)
        h.execute("hold_seat", dict(flight_id="VN101", quoted_price=1_200_000))
        h.execute("check_seat", dict(flight_id="VJ202"))
        r = h.execute("hold_seat", dict(flight_id="VJ202", quoted_price=1_400_000))
        self.assertEqual(r["error_code"], "ACTIVE_BOOKING_EXISTS")
        self.assertEqual(len(h.backend.bookings), 1)

    def test_backend_writes_idempotent(self):
        b = MockAirline(scenarios()["normal"])
        args = dict(flight_id="VN101", quoted_price=1_200_000)
        bid = b.execute("hold_seat", args)["booking"]["booking_id"]
        b.execute("hold_seat", args)
        b.execute("pay", dict(booking_id=bid))
        b.execute("pay", dict(booking_id=bid))
        self.assertEqual(b.flights["VN101"]["seats"], 1)
        self.assertEqual(len(b.audit), 2)
        self.assertEqual(b.charges, {"BK001": 1_200_000})

    def test_schema_errors_stop_before_execution(self):
        h = self.make()
        r = h.execute("pay", dict(booking_id="x", unexpected=True))
        self.assertEqual(r["error_code"], "INVALID_ARGS")
        self.assertEqual(h.tool_calls, 0)

    def test_stall_without_repeated_action(self):
        h = self.make(stall_n=3)
        self.search(h)
        for i in range(3):
            h.execute("check_seat", dict(flight_id=f"UNKNOWN-{i}"))
        self.assertEqual(h.status, "STALL")

    def test_wall_time_budget(self):
        h = self.make(seconds=0)
        self.search(h)
        self.assertEqual(h.status, "BUDGET_EXCEEDED")
        self.assertEqual(h.tool_calls, 0)

    def test_payment_without_readback_is_not_success(self):
        h = self.make()
        self.prepare(h)
        h.execute("hold_seat", dict(flight_id="VN101", quoted_price=1_200_000))
        h.execute("pay", dict(booking_id="BK001"))
        h.finish()
        self.assertEqual(h.status, "UNVERIFIED_COMPLETION")


if __name__ == "__main__":
    unittest.main(verbosity=2)
