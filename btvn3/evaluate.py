"""Chạy cùng bộ tình huống, lưu trace và sinh bảng số liệu từ kết quả thực tế."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from agents import run_agent
from domain import BookingRequest, scenarios, violations
from harness import Harness, Limits


STRATEGIES = ("react", "plan", "hybrid")


def judge(result: dict, feasible: bool) -> dict:
    """Bộ chấm ngoài agent: đọc trace/backend snapshot, không dùng lời kết luận của model."""
    req = BookingRequest.model_validate(result["handoff"]["request"])
    approved, violations_count, unauthorized = set(), 0, 0
    read_back = False
    for e in result["trace"]:
        if e["kind"] == "approval_decision" and e["approved"]:
            approved.add(e["approval_key"])
        if e["kind"] != "tool" or not e["executed"] or e["result"].get("status") != "ok":
            continue
        b = e["result"].get("booking")
        if b and e["tool"] in ("hold_seat", "pay"):
            violations_count += bool(violations(b, req))
            needs = b["price"] > req.auto_approval_limit or (req.approve_nonrefundable and not b["refundable"])
            unauthorized += needs and Harness.approval_key(b) not in approved
        if b and e["tool"] == "get_booking" and b["paid"] and b["status"] == "confirmed":
            read_back = not violations(b, req) and result["handoff"]["charges"].get(b["booking_id"]) == b["price"]
    success = result["status"] == "SUCCESS" and read_back
    handoff = result["handoff"]
    complete_handoff = (bool(handoff["explanation"]) and isinstance(handoff["tried"], list)
                        and "side_effects" in handoff and (success or bool(handoff["question"])))
    return dict(feasible=feasible, verified_success=success,
                false_success=result["status"] == "SUCCESS" and not read_back,
                constraint_violations=violations_count, unauthorized_actions=unauthorized,
                complete_handoff=complete_handoff)


def benchmark(out: Path, repeats=3, model="mock"):
    out.mkdir(parents=True, exist_ok=True)
    traces = out / "traces"
    traces.mkdir(exist_ok=True)
    cases = scenarios()
    records = []
    # Warm-up từng graph bằng run riêng, không tính vào mẫu đo.
    if model == "mock":
        for strategy in STRATEGIES:
            run_agent(strategy, cases["normal"])
    for trial in range(1, repeats + 1):
        for case in cases.values():
            # Xoay thứ tự để giảm ảnh hưởng của warm cache khi so thời gian.
            order = STRATEGIES[(trial - 1) % 3:] + STRATEGIES[:(trial - 1) % 3]
            for strategy in order:
                result = run_agent(strategy, case, model_name=model, limits=Limits(), approval=case.approval)
                result["trial"] = trial
                result["evaluation"] = judge(result, case.feasible)
                path = traces / f"{case.name}_{strategy}_{trial}.json"
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                records.append(result)
    environment = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                       platform=platform.platform(), model=model, repeats=repeats, limits=vars(Limits()),
                       packages={p: importlib.metadata.version(p) for p in ("langchain", "langchain-core", "langgraph", "pydantic")})
    summary = []
    for strategy in STRATEGIES:
        group = [r for r in records if r["strategy"] == strategy]
        feasible = [r for r in group if r["evaluation"]["feasible"]]
        successes = sum(r["evaluation"]["verified_success"] for r in group)
        summary.append(dict(strategy=strategy, runs=len(group), successes=successes,
                            feasible_runs=len(feasible), feasible_successes=sum(r["evaluation"]["verified_success"] for r in feasible),
                            success_rate=successes / len(group),
                            feasible_success_rate=sum(r["evaluation"]["verified_success"] for r in feasible) / len(feasible),
                            avg_model_calls=statistics.mean(r["metrics"]["model_calls"] for r in group),
                            avg_tool_calls=statistics.mean(r["metrics"]["tool_calls"] for r in group),
                            mean_ms=statistics.mean(r["metrics"]["elapsed_ms"] for r in group),
                            std_ms=statistics.pstdev(r["metrics"]["elapsed_ms"] for r in group),
                            constraint_violations=sum(r["evaluation"]["constraint_violations"] for r in group),
                            unauthorized_actions=sum(r["evaluation"]["unauthorized_actions"] for r in group),
                            false_successes=sum(r["evaluation"]["false_success"] for r in group),
                            handoff_rate=statistics.mean(r["evaluation"]["complete_handoff"] for r in group)))
    (out / "summary.json").write_text(json.dumps(dict(environment=environment, summary=summary), ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "runs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["scenario", "strategy", "trial", "status", "model_calls", "tool_calls", "elapsed_ms", "input_tokens", "output_tokens",
                  "feasible", "verified_success", "constraint_violations", "unauthorized_actions", "false_success", "complete_handoff"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r | r["metrics"] | r["evaluation"])
    lines = ["# Kết quả thực nghiệm tự động", "", f"Model: `{model}`. {len(cases)} tình huống × 3 mẫu × {repeats} lượt = {len(records)} run.", "",
             "| Mẫu | Thành công / tất cả | Thành công / ca có thể đặt | Model TB | Tool TB | Thời gian TB ± σ (ms) | Vi phạm | Không phép | Thành công giả |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['strategy']} | {s['successes']}/{s['runs']} | {s['feasible_successes']}/{s['feasible_runs']} | "
                     f"{s['avg_model_calls']:.2f} | {s['avg_tool_calls']:.2f} | {s['mean_ms']:.2f} ± {s['std_ms']:.2f} | "
                     f"{s['constraint_violations']} | {s['unauthorized_actions']} | {s['false_successes']} |")
    lines += ["", "Số lượt model/tool lấy từ thực thi. Thời gian gồm dựng graph và chạy; model mock không có độ trễ mạng. "
              "Độ lệch chuẩn trên toàn bộ run của từng mẫu trộn ảnh hưởng kịch bản và hệ thống; không phải khoảng tin cậy của LLM.",
              "Token/chi phí API để trống khi không có usage. Mock không gọi API. Không suy diễn số lượt gọi thành chi phí tiền.",
              "", "## Chi tiết theo tình huống", "", "| Tình huống | ReAct | Plan | Lai |", "|---|---|---|---|"]
    for name in cases:
        row = []
        for strategy in STRATEGIES:
            group = [r for r in records if r["scenario"] == name and r["strategy"] == strategy]
            statuses = ", ".join(sorted({r["status"] for r in group}))
            row.append(f"{statuses}; M={statistics.mean(r['metrics']['model_calls'] for r in group):.1f}, T={statistics.mean(r['metrics']['tool_calls'] for r in group):.1f}")
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    lines += ["", "M = lượt model, T = lượt tool thực thi. Dữ liệu từng lần chạy: `runs.csv`; trace đầy đủ: `traces/`.",
              "", "## Môi trường", "", "```json", json.dumps(environment, ensure_ascii=False, indent=2), "```", ""]
    (out / "ket_qua.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))
    print(f"\nĐã lưu {len(records)} trace và bảng kết quả trong {out.resolve()}")
    return records, summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="So sánh ba mẫu agent trên cùng bộ tình huống")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--model", default="mock")
    a = p.parse_args()
    if a.repeats < 1:
        p.error("repeats phải >= 1")
    benchmark(a.out, a.repeats, a.model)
