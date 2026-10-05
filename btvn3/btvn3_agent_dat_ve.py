"""Điểm chạy BTVN#3. Xem README.md để chạy demo, benchmark và test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents import run_agent
from domain import BookingRequest, scenarios
from harness import Limits, as_json


def print_result(result: dict, trace=False):
    print(f"\n{result['strategy'].upper()} | {result['scenario']} | {result['status']}")
    print(result["handoff"]["explanation"])
    m = result["metrics"]
    print(f"Model: {m['model_calls']} lượt | Tool: {m['tool_calls']} lượt | {m['elapsed_ms']:.1f} ms")
    b = result["verified_booking"]
    if b:
        print(f"Vé {b['booking_id']}: {b['flight_id']} | {b['depart_date']} {b['departure']} | {b['price']:,} VND")
    else:
        print("Bàn giao:", as_json(result["handoff"]))
    if trace:
        for e in result["trace"]:
            print(as_json(e))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="SE373 BTVN#3: agent đặt vé giả lập")
    p.add_argument("--strategy", choices=["react", "plan", "hybrid"], default="react")
    p.add_argument("--scenario", choices=list(scenarios()), default="normal")
    p.add_argument("--model", default="mock", help="mock hoặc provider:model; cài integration tương ứng để dùng LLM thật")
    p.add_argument("--approval", choices=["ask", "approve", "reject", "defer"], default="ask")
    p.add_argument("--request", type=Path, help="JSON ràng buộc; mặc định SGN-DAD sáng 07/10/2026, tối đa 2 triệu")
    p.add_argument("--max-model-calls", type=int, default=30)
    p.add_argument("--max-tool-calls", type=int, default=30)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--chunk-size", type=int, default=2, help="Số bước tối đa trước khi mẫu Lai lập lại kế hoạch")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--output", type=Path, help="Lưu đầy đủ kết quả/trace dạng JSON")
    p.add_argument("--list-scenarios", action="store_true")
    args = p.parse_args()
    if args.list_scenarios:
        for s in scenarios().values():
            print(f"{s.name:22} {s.description}")
        return 0
    if min(args.max_model_calls, args.max_tool_calls, args.chunk_size) <= 0 or args.seconds <= 0:
        p.error("Giới hạn và chunk-size phải lớn hơn 0.")
    req = BookingRequest.model_validate_json(args.request.read_text(encoding="utf-8")) if args.request else BookingRequest()
    result = run_agent(args.strategy, scenarios()[args.scenario], model_name=args.model, request=req,
                       approval=args.approval, chunk_size=args.chunk_size,
                       limits=Limits(model_calls=args.max_model_calls, tool_calls=args.max_tool_calls, seconds=args.seconds))
    print_result(result, args.trace)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Đã lưu:", args.output)
    return 1 if result["status"] == "EXECUTION_ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
