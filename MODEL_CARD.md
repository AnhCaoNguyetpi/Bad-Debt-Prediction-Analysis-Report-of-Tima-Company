# Model card template

Điền tài liệu này sau mỗi lần huấn luyện chính thức bằng thông tin trong
`artifacts/metrics.json`.

## Mục đích

- Bài toán: dự báo xác suất nợ xấu trước giải ngân.
- Người dùng dự kiến: nhóm phân tích/rủi ro tín dụng.
- Không dùng để tự động từ chối khách hàng nếu chưa có phê duyệt quản trị và pháp lý.

## Dữ liệu

- Phiên bản/khoảng thời gian: **Cần điền**
- Định nghĩa `HasBadDebt`: **Cần điền**
- Observation/performance window: **Cần điền**
- Số dòng, tỷ lệ nợ xấu: lấy từ `metrics.json`
- Các nhóm bị thiếu hoặc đại diện kém: **Cần điền**

## Kết quả

- Model/ngưỡng được chọn: lấy từ `metadata`
- PR-AUC, ROC-AUC, precision, recall, F1 và confusion matrix: lấy từ `test_metrics`
- Baseline nghiệp vụ: **Cần điền**
- Đánh giá out-of-time: **Cần điền**
- Đánh giá fairness và calibration: **Cần điền**

## Hạn chế và giám sát

- Kết quả phụ thuộc chất lượng nhãn và tính đúng thời điểm của feature.
- Correlation/SHAP không chứng minh quan hệ nhân quả.
- Theo dõi tỷ lệ thiếu, PSI/drift, calibration, PR-AUC và recall theo tháng.
- Đặt lịch tái thẩm định và tiêu chí rollback: **Cần điền**
