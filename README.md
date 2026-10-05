# BTVN#3 — Agent đặt vé máy bay bằng LangChain

Bài tập SE373: cài **ReAct**, **Plan-then-Execute** và **mẫu Lai** với cùng tool mockup và harness.
Mặc định chạy bằng model giả lập, không cần API key; mọi thao tác vé và thanh toán đều là giả lập.

## Chạy trên Windows / PowerShell

```powershell
git clone https://github.com/tranthang23521432/btvn-3.git
cd btvn-3
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r btvn3/requirements.txt

# ReAct: ca bình thường
.\.venv\Scripts\python.exe -X utf8 btvn3/btvn3_agent_dat_ve.py --strategy react --scenario normal

# So sánh kế hoạch cố định và mẫu Lai khi hết ghế
.\.venv\Scripts\python.exe -X utf8 btvn3/btvn3_agent_dat_ve.py --strategy plan --scenario sold_out --trace
.\.venv\Scripts\python.exe -X utf8 btvn3/btvn3_agent_dat_ve.py --strategy hybrid --scenario sold_out --trace

# Kiểm thử và đo lại
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s btvn3 -p test_btvn3.py -v
.\.venv\Scripts\python.exe -X utf8 btvn3/evaluate.py --repeats 3
```

Trên Linux/macOS, dùng `.venv/bin/python` thay cho `.\.venv\Scripts\python.exe`.
Đã kiểm thử bằng Python 3.14.5; phiên bản dependency được khóa trong `btvn3/requirements.txt`.

## Nội dung bài nộp

- [Báo cáo](btvn3/bao_cao.md): thiết kế harness, kiến trúc, phương pháp đánh giá và phân tích trace.
- [Hướng dẫn chi tiết](btvn3/README.md): demo phê duyệt, cấu hình request, giới hạn và tùy chọn model thật.
- [Điểm chạy chương trình](btvn3/btvn3_agent_dat_ve.py).
- [Harness dùng chung](btvn3/harness.py), [ba mẫu agent](btvn3/agents.py), [tool và kịch bản](btvn3/domain.py).
- [Bộ kiểm thử](btvn3/test_btvn3.py) và [mã đánh giá](btvn3/evaluate.py).
- [Kết quả thực nghiệm](btvn3/results/ket_qua.md), [dữ liệu từng lượt](btvn3/results/runs.csv), [108 trace](btvn3/results/traces).

## Kết quả trên model giả lập

12 tình huống × 3 mẫu × 3 lượt = **108 run**. Đã qua **22 test method**, gồm ma trận cả 36 cặp mẫu/tình huống.

| Mẫu | Thành công trên ca có thể đặt | Model calls trung bình | Vi phạm ràng buộc / vượt quyền |
|---|---:|---:|---:|
| ReAct | 24/24 | 4,92 | 0 / 0 |
| Plan-then-Execute | 12/24 | 1,00 | 0 / 0 |
| Lai | 24/24 | 2,75 | 0 / 0 |

Plan là baseline kế hoạch cố định: khi bước thất bại, hệ thống dừng và bàn giao. Lai lập lại kế hoạch để thích nghi.
Kết quả chỉ đánh giá bản triển khai và chính sách giả lập này; chưa phải số liệu về chất lượng suy luận hoặc chi phí của LLM thật.

Harness lưu ràng buộc bằng dữ liệu, kiểm hoàn thành bằng code, chặn trước hành động cần duyệt,
và bàn giao trạng thái cùng tác dụng phụ khi chưa hoàn thành. Checkpoint hiện ở bộ nhớ, chỉ resume trong cùng tiến trình.

Thông tin họ tên/MSSV trong báo cáo được để trống để người nộp điền chính xác.
