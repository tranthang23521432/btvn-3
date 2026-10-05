# Kết quả thực nghiệm tự động

Model: `mock`. 12 tình huống × 3 mẫu × 3 lượt = 108 run.

| Mẫu | Thành công / tất cả | Thành công / ca có thể đặt | Model TB | Tool TB | Thời gian TB ± σ (ms) | Vi phạm | Không phép | Thành công giả |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| react | 24/36 | 24/24 | 4.92 | 4.67 | 15.57 ± 3.93 | 0 | 0 | 0 |
| plan | 12/36 | 12/24 | 1.00 | 3.17 | 7.38 ± 1.41 | 0 | 0 | 0 |
| hybrid | 24/36 | 24/24 | 2.75 | 4.67 | 11.70 ± 5.43 | 0 | 0 | 0 |

Số lượt model/tool lấy từ thực thi. Thời gian gồm dựng graph và chạy; model mock không có độ trễ mạng. Độ lệch chuẩn trên toàn bộ run của từng mẫu trộn ảnh hưởng kịch bản và hệ thống; không phải khoảng tin cậy của LLM.
Token/chi phí API để trống khi không có usage. Mock không gọi API. Không suy diễn số lượt gọi thành chi phí tiền.

## Chi tiết theo tình huống

| Tình huống | ReAct | Plan | Lai |
|---|---|---|---|
| normal | SUCCESS; M=5.0, T=5.0 | SUCCESS; M=1.0, T=5.0 | SUCCESS; M=3.0, T=5.0 |
| sold_out | SUCCESS; M=6.0, T=6.0 | PLAN_FAILED; M=1.0, T=2.0 | SUCCESS; M=3.0, T=6.0 |
| price_jump | SUCCESS; M=6.0, T=6.0 | PLAN_FAILED; M=1.0, T=2.0 | SUCCESS; M=3.0, T=6.0 |
| no_match | NO_SUITABLE_FLIGHT; M=2.0, T=1.0 | NO_SUITABLE_FLIGHT; M=1.0, T=1.0 | NO_SUITABLE_FLIGHT; M=1.0, T=1.0 |
| approval_yes | SUCCESS; M=5.0, T=5.0 | SUCCESS; M=1.0, T=5.0 | SUCCESS; M=3.0, T=5.0 |
| approval_no | APPROVAL_REJECTED; M=3.0, T=2.0 | APPROVAL_REJECTED; M=1.0, T=2.0 | APPROVAL_REJECTED; M=2.0, T=2.0 |
| approval_wait | NEEDS_APPROVAL; M=3.0, T=2.0 | NEEDS_APPROVAL; M=1.0, T=2.0 | NEEDS_APPROVAL; M=2.0, T=2.0 |
| transient_error | SUCCESS; M=6.0, T=6.0 | PLAN_FAILED; M=1.0, T=2.0 | SUCCESS; M=3.0, T=6.0 |
| persistent_error | LOOP_DETECTED; M=4.0, T=4.0 | PLAN_FAILED; M=1.0, T=2.0 | LOOP_DETECTED; M=3.0, T=4.0 |
| wrong_constraints | SUCCESS; M=5.0, T=5.0 | SUCCESS; M=1.0, T=5.0 | SUCCESS; M=3.0, T=5.0 |
| confirmation_delay | SUCCESS; M=7.0, T=7.0 | SUCCESS; M=1.0, T=7.0 | SUCCESS; M=3.0, T=7.0 |
| hold_race | SUCCESS; M=7.0, T=7.0 | PLAN_FAILED; M=1.0, T=3.0 | SUCCESS; M=4.0, T=7.0 |

M = lượt model, T = lượt tool thực thi. Dữ liệu từng lần chạy: `runs.csv`; trace đầy đủ: `traces/`.

## Môi trường

```json
{
  "created_utc": "2026-10-05T18:16:20.981823+00:00",
  "python": "3.14.5",
  "platform": "Windows-11-10.0.26200-SP0",
  "model": "mock",
  "repeats": 3,
  "limits": {
    "model_calls": 30,
    "tool_calls": 30,
    "seconds": 60.0,
    "repeat_k": 3,
    "stall_n": 8,
    "poll_limit": 4,
    "replans": 8
  },
  "packages": {
    "langchain": "1.4.3",
    "langchain-core": "1.6.6",
    "langgraph": "1.2.13",
    "pydantic": "2.13.5"
  }
}
```
