# Kế hoạch Phát triển Dự án: Hệ thống Gợi ý Sản phẩm (Recommendation System)

## 1. Tổng quan dự án
Xây dựng hệ thống gợi ý sản phẩm dựa trên hành vi người dùng (Click, Mua, Xem) sử dụng các thuật toán Collaborative Filtering và Content-based Filtering. Dự án giúp tối ưu hóa doanh thu và tăng trải nghiệm người dùng trong thương mại điện tử.

## 2. Kiến trúc Hệ thống (Architecture)
Dự án được chia làm hai luồng chính:
- **Offline Processing**: 
  - Thu thập dữ liệu lịch sử (Data).
  - Trích xuất đặc trưng (Feature Extraction).
  - Huấn luyện mô hình (Model Training) và Đánh giá (Model Testing).
- **Online Processing**:
  - Nhận yêu cầu từ người dùng thực tế.
  - Chạy mô hình để Ranking (xếp hạng) các sản phẩm tiềm năng.
  - Trả về danh sách gợi ý (Rendering) và ghi lại nhật ký (Logging).

## 3. Tech Stack Đề xuất
- **Ngôn ngữ**: Python (chủ đạo cho Data Science).
- **Thư viện AI/ML**: Scikit-learn, Surprise, hoặc TensorFlow/PyTorch.
- **Xử lý dữ liệu**: Pandas, NumPy.
- **Database**: PostgreSQL (dữ liệu thô) và Redis (cache kết quả gợi ý).
- **Frontend**: React.js để hiển thị danh sách sản phẩm.
- **Dataset**: MovieLens hoặc các dataset E-commerce từ Kaggle.

## 4. Lộ trình thực hiện (Timeline)

### Tuần 1: Thu thập & Tiền xử lý dữ liệu ✅
- [x] Chọn Dataset thực tế — **MovieLens ml-latest-small** (610 users, 2269 movies, 81K ratings).
- [x] Xử lý dữ liệu khuyết thiếu, chuẩn hóa định dạng dữ liệu User-Item → `data/processed/ratings_clean.csv`, `user_item_matrix.csv`.
- [x] Thực hiện Exploratory Data Analysis (EDA) — biểu đồ rating distribution, top movies, genre frequency lưu tại `notebooks/figures/`.

### Tuần 2: Xây dựng Mô hình gợi ý (Core Logic) ✅
- [x] Triển khai **Collaborative Filtering** (`src/collaborative/`):
  - `user_based.py` — Cosine similarity giữa users, weighted score aggregation.
  - `item_based.py` — Cosine similarity giữa items, rating-weighted scoring.
- [x] Triển khai **Content-based Filtering** (`src/content_based/content_filter.py`): TF-IDF trên genre → Cosine similarity giữa các phim.

### Tuần 3: Tối ưu hóa & Ranking ✅
- [x] Kết hợp các mô hình **Hybrid Approach** (`src/hybrid/hybrid_recommender.py`): weights `user_cf=0.4, item_cf=0.3, content=0.3`, normalize scores về [0,1] trước khi blend.
- [x] Luồng Scoring và Ranking: Top-N selection sau khi cộng điểm từ 3 nguồn.
- [x] Đánh giá mô hình (`src/evaluate.py`): Precision@K, Recall@K với leave-one-out. *(P@K ≈ 0 là bình thường với catalog 2000+ items — xem ghi chú trong file).*

### Tuần 4: Triển khai API & Giao diện ✅
- [x] Viết API bằng **FastAPI** (`src/api/main.py`):
  - `GET /recommend/{user_id}?n=10` — Top-N hybrid recommendations.
  - `GET /similar/{movie_id}?n=10` — Content-based similar movies.
  - `GET /health` — Health check.
- [ ] Xây dựng giao diện Web đơn giản hiển thị "Sản phẩm dành cho bạn" *(React — chưa thực hiện)*.
- [x] Đóng gói dự án bằng **Docker** (`Dockerfile`, `docker-compose.yml`).

## 5. Các kỹ thuật cần nắm vững
- Matrix Factorization (SVD).
- Cosine Similarity / Pearson Correlation.
- Cold Start Problem (Cách xử lý khi người dùng mới chưa có dữ liệu).

## 6. Cấu trúc thư mục
```
Recommendation-System/
├── data/
│   ├── raw/ml-latest-small/     # Dataset gốc (MovieLens)
│   └── processed/               # ratings_clean.csv, user_item_matrix.csv
├── notebooks/figures/           # Biểu đồ EDA
├── src/
│   ├── data_preprocessing.py    # Tiền xử lý dữ liệu
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── evaluate.py              # Precision@K, Recall@K
│   ├── collaborative/
│   │   ├── user_based.py        # User-based CF
│   │   └── item_based.py        # Item-based CF
│   ├── content_based/
│   │   └── content_filter.py    # TF-IDF Content-based
│   ├── hybrid/
│   │   └── hybrid_recommender.py # Weighted Hybrid
│   └── api/
│       └── main.py              # FastAPI endpoints
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 7. Cách chạy

```bash
# Cài thư viện
pip install -r requirements.txt

# Tiền xử lý dữ liệu
python src/data_preprocessing.py

# Chạy API
uvicorn src.api.main:app --reload

# Hoặc chạy bằng Docker
docker-compose up --build
```
