# Dự báo nợ xấu – Tima Company

Project xây dựng mô hình phân loại rủi ro nợ xấu từ dữ liệu hồ sơ vay. Luồng chạy
chính được thiết kế cho **thời điểm trước giải ngân**, có kiểm soát rò rỉ dữ liệu,
tách khách hàng giữa các tập dữ liệu và đánh giá trên tập test độc lập.

> Notebook và Power BI trong repository là tài liệu phân tích lịch sử. Kết quả mô
> hình chính thức phải được tạo lại bằng `train_model.py`; không sử dụng các bảng
> metric hard-code trong notebook làm kết quả báo cáo.

## Cấu trúc

| File | Vai trò |
|---|---|
| `train_model.py` | Đọc/kiểm tra dữ liệu, tạo feature an toàn, chia dữ liệu, chọn mô hình và ngưỡng, đánh giá, lưu artifacts. |
| `predict.py` | Chấm điểm dữ liệu mới bằng model đã lưu. |
| `DATA_DICTIONARY.md` | Hợp đồng dữ liệu và lưu ý quản trị. |
| `processing_data.ipynb` | Notebook xử lý dữ liệu cũ, chỉ dùng tham khảo. |
| `exploration.ipynb` | EDA lịch sử. |
| `Building_predictive_model.ipynb` | Thử nghiệm mô hình lịch sử, không phải pipeline chuẩn. |
| `Dashboard.pbix` | Dashboard Power BI. |

## Thiết kế mô hình

- Mặc định loại định danh cá nhân và các biến hậu nghiệm như `HasLatePayment`,
  `LongestOverdue`, `TienGiaiNgan`, `SoTienConLai` và trạng thái thanh toán.
- Tạo `Age` theo ngày tham chiếu cố định và `LoanToIncome` một cách an toàn.
- Imputation, scaling và one-hot encoding đều được fit **chỉ trên train**.
- Nếu có mã khách hàng, chia train/validation/test theo khách hàng (60/20/20), tránh
  một khách hàng xuất hiện ở nhiều tập. Nếu không có, dùng stratified split.
- So sánh Logistic Regression và Random Forest bằng PR-AUC trên validation.
- Chọn classification threshold trên validation; test chỉ dùng để báo cáo cuối.
- Báo cáo precision/recall/F1 của lớp nợ xấu, balanced accuracy, ROC-AUC, PR-AUC và
  confusion matrix.

## Chuẩn bị dữ liệu

Đặt CSV vào `data/raw/` (thư mục này bị Git bỏ qua để tránh lộ dữ liệu khách hàng).
CSV cần có `HasBadDebt` với giá trị 0/1 và tối thiểu 50 dòng, trong đó mỗi lớp có ít
nhất 10 dòng. Xem chi tiết tại [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

Nếu dữ liệu có cột nhận diện khách hàng, nên chỉ định cột đó bằng `--group-column`.
Không được đưa dữ liệu cá nhân thật lên GitHub.

## Cài đặt trên Windows PowerShell

Chạy từ thư mục project:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra code trước khi huấn luyện:

```powershell
python -m unittest -v test_pipeline.py
python train_model.py --help
python predict.py --help
```

## Huấn luyện

Lệnh khuyến nghị, thay tên cột khách hàng cho đúng dữ liệu của bạn:

```powershell
python train_model.py `
  --input ".\data\raw\dataset.csv" `
  --group-column "CustomerID" `
  --as-of-date "2019-03-17"
```

Nếu không có cột định danh khách hàng:

```powershell
python train_model.py --input ".\data\raw\dataset.csv"
```

Loại thêm các cột chỉ có sau quyết định cho vay hoặc có nguy cơ tiết lộ target:

```powershell
python train_model.py `
  --input ".\data\raw\dataset.csv" `
  --exclude-columns "CollectionResult" "WriteOffDate" "RecoveryAmount"
```

Trong trường hợp ưu tiên bắt được nhiều khoản nợ xấu hơn và chấp nhận false positive:

```powershell
python train_model.py `
  --input ".\data\raw\dataset.csv" `
  --threshold-objective recall `
  --min-precision 0.25
```

`--include-post-outcome` chỉ dành cho bài toán giám sát sau giải ngân, tuyệt đối không
dùng kết quả đó để tuyên bố hiệu năng chấm điểm trước giải ngân.

## Kết quả sinh ra

Sau khi chạy thành công, thư mục `artifacts/` gồm:

- `bad_debt_model.joblib`: model cùng preprocessing, threshold và metadata;
- `metrics.json`: leaderboard validation và kết quả test cuối;
- `evaluation.png`: confusion matrix, Precision–Recall và ROC curve;
- `test_predictions.csv`: xác suất và dự báo trên tập test.

Khi đọc kết quả, ưu tiên `test_metrics.pr_auc`, `recall_bad_debt`,
`precision_bad_debt` và confusion matrix. Accuracy cao không đủ chứng minh mô hình tốt
khi tỷ lệ nợ xấu thấp.

## Dự báo dữ liệu mới

File mới phải có đầy đủ các feature được ghi trong metadata model; không cần cột
`HasBadDebt`.

```powershell
python predict.py `
  --input ".\data\raw\new_applications.csv" `
  --output ".\artifacts\predictions.csv"
```

Output bổ sung `ProbabilityBadDebt`, `PredictedBadDebt` và `RiskBand`. Risk band chỉ
là nhóm trình bày; quyết định tín dụng phải dựa trên ngưỡng, chi phí kinh doanh,
fairness, quy định pháp lý và phê duyệt của chuyên gia rủi ro.

## Checklist trước khi công bố

1. Chốt định nghĩa nợ xấu và cửa sổ quan sát target.
2. Xác nhận từng feature tồn tại tại đúng thời điểm ra quyết định.
3. Kiểm tra đơn vị tiền, tỷ lệ thiếu, outlier và drift theo thời gian.
4. Nếu có ngày hồ sơ, bổ sung đánh giá out-of-time trước khi triển khai thật.
5. Đánh giá fairness theo nhóm nhạy cảm và hiệu chỉnh xác suất.
6. Ghi lại phiên bản dữ liệu/code cùng `metrics.json`; không chọn model dựa trên test.

Power BI là tùy chọn và chỉ cần để mở `Dashboard.pbix`.
