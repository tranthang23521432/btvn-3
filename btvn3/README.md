# SE373 — BTVN#3: Agent đặt vé máy bay bằng LangChain

Bộ bài tập gồm ba kiến trúc ReAct, Plan-then-Execute và Lai, cùng một backend vé giả lập và harness.
Mặc định chạy hoàn toàn trên máy, không cần API key, không đặt vé hay thanh toán thật.

## 1. Chạy nhanh trên máy hiện tại

Mở PowerShell tại thư mục `Demo tại lớp-20260928`. Môi trường `.venv` đã được cài và kiểm thử.

```powershell
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --strategy react --scenario normal
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --strategy plan --scenario sold_out --trace
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --strategy hybrid --scenario sold_out --trace
```

Lệnh thứ hai kết thúc `PLAN_FAILED` là kết quả có chủ đích của baseline kế hoạch cố định khi chuyến đã chọn hết ghế.
Mẫu Lai lập lại kế hoạch để thử chuyến khác.

Demo người phê duyệt thật qua bàn phím:

```powershell
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --strategy hybrid --scenario approval_yes --approval ask
```

Nhập `yes` để tiếp tục, `no` để từ chối, Enter để xuất bàn giao `NEEDS_APPROVAL`.
Nội dung xin duyệt nêu rõ phạm vi giữ chỗ VÀ thanh toán, gắn với đúng chuyến, giá và điều kiện đang hiển thị.
LangGraph dùng `interrupt()` và `Command(resume=...)` để tiếp tục ngay trong cùng tiến trình.
Checkpointer là `InMemorySaver`: kết thúc tiến trình sẽ mất checkpoint. File kết quả JSON là báo cáo,
không phải checkpoint để resume ở lần khởi động CLI khác.

## 2. Cài và chạy trên máy khác

Từ bên trong thư mục `btvn3`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 btvn3_agent_dat_ve.py --strategy react --scenario normal
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s . -p test_btvn3.py -v
```

Môi trường đã kiểm chứng: Python 3.14.5, LangChain 1.4.3, LangGraph 1.2.13,
langchain-core 1.6.6, Pydantic 2.13.5. Có `requirements-lock.txt` chứa toàn bộ dependency đã cài
để đối chiếu/tái lập; `requirements.txt` khóa các dependency trực tiếp.

## 3. Đánh giá lại

Từ thư mục cha như mục 1:

```powershell
.\.venv\Scripts\python.exe -X utf8 btvn3\evaluate.py --repeats 3
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s btvn3 -p test_btvn3.py -v
```

Đánh giá tạo 12 tình huống × 3 mẫu × 3 lần = 108 kết quả:

- `results/ket_qua.md`: bảng số liệu dễ đọc, sinh từ lần chạy.
- `results/summary.json`: chỉ số tổng hợp và môi trường.
- `results/runs.csv`: dữ liệu từng lượt.
- `results/traces/*.json`: toàn bộ bước model/tool, phê duyệt và bàn giao.

Khi chạy lại mặc định sẽ cập nhật `results/`. Dùng `--out btvn3/results_lan_2` nếu muốn giữ bộ cũ.
Báo cáo `bao_cao.md` ghi số liệu của bộ kết quả nộp; nếu chạy lại, cần đồng bộ bảng thời gian trong báo cáo.

## 4. Tham số hữu ích

```powershell
# Danh sách tình huống
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --list-scenarios

# Lưu một trace
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --strategy react --scenario persistent_error --trace --output btvn3/demo_trace.json

# Chặn theo ngân sách
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --strategy react --scenario normal --max-tool-calls 2

# Đọc ràng buộc từ JSON
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --request btvn3/request.example.json
```

`--approval approve|reject|defer` dùng cho demo tự động. Các thao tác vẫn là giả lập.
Benchmark cấp cùng quyết định phê duyệt từ bên ngoài model cho ba mẫu.
`max_price` là trần cứng; phê duyệt không làm tăng trần này. `auto_approval_limit` là hạn mức được làm tự động.
Thay đổi yêu cầu phải khởi tạo phiên mới với dữ liệu yêu cầu mới.

## 5. Dùng LLM thật (tùy chọn, chưa đo trong bản nộp)

`--model mock` dùng một chính sách Python qua `BaseChatModel`, tương tự model giả trong demo trên lớp.
Để dùng model thật, cài gói integration của nhà cung cấp, thiết lập thông tin xác thực theo tài liệu của họ,
rồi truyền model có hỗ trợ tool calling ở dạng `provider:model-name`:

```powershell
# Thay provider:model-name bằng model đã cấu hình trên máy của bạn
.\.venv\Scripts\python.exe -X utf8 btvn3\btvn3_agent_dat_ve.py --model 'provider:model-name' --strategy react --scenario normal
.\.venv\Scripts\python.exe -X utf8 btvn3\evaluate.py --model 'provider:model-name' --repeats 3 --out btvn3/results_real
```

Model thật được tạo bằng `init_chat_model`, đặt temperature=0, timeout=30 giây, max_retries=0;
khả năng hỗ trợ tham số phụ thuộc integration. Kế hoạch được yêu cầu trả JSON và kiểm bằng Pydantic.
Không đưa API key vào code hay bài nộp. Chưa chạy chế độ này trong bản nộp vì chưa cấu hình model/API;
token và chi phí không được suy ra từ số lời gọi. Nếu provider trả `usage_metadata`, trace sẽ lưu usage.

## 6. Đọc code theo thứ tự

1. `domain.py`: ràng buộc, schema tool, 12 tình huống và trạng thái vé.
2. `harness.py`: kiểm ràng buộc/quyền, phát hiện lặp/bế tắc, giới hạn, bàn giao.
3. `agents.py`: model giả, ReAct với `create_agent`, Plan và Lai với `StateGraph`.
4. `btvn3_agent_dat_ve.py`: giao diện dòng lệnh.
5. `evaluate.py`: đo và chấm kết quả bên ngoài model.
6. `test_btvn3.py`: kiểm thử nghiệp vụ và các trường hợp model làm sai.

## 7. Nội dung nộp và trình bày

Nộp toàn bộ thư mục `btvn3` (các module `.py`, báo cáo, requirements, request mẫu và results),
không cần `.venv` hoặc `__pycache__`. Có thể tải ZIP từ nút Code trên repository GitHub.
Điền tên, MSSV và nhóm vào đầu `bao_cao.md` trước khi nộp.

Kịch bản trình bày khoảng 5 phút:

1. Chạy `normal`: chỉ ra tool trả tiền khác với bước đọc lại xác nhận thành công.
2. Chạy `sold_out` cho Plan và Lai: mở trace tại bước `check_seat` và lần lập lại kế hoạch.
3. Chạy `approval_yes --approval ask`, nhập `no`: chứng minh không có giữ chỗ/thanh toán sau từ chối.
4. Chạy `persistent_error`: chỉ ra ba lần timeout và lý do dừng.
5. Mở bảng `results/ket_qua.md`, giải thích phạm vi kết luận của model giả.
