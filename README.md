# Recommendation System

Hệ thống gợi ý sản phẩm sử dụng Collaborative Filtering, Content-based Filtering và Hybrid Approach, xây dựng trên dataset MovieLens.

## Kiến trúc

```
User Request
    │
    ▼
FastAPI (/recommend/{user_id})
    │
    ├── User-based CF  (40%)
    ├── Item-based CF  (30%)
    └── Content-based  (30%)
            │
            ▼
     Weighted Hybrid Score → Top-N Results
```

## Tech Stack

| Layer | Công nghệ |
|---|---|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| ML | Scikit-learn (Cosine Similarity, TF-IDF) |
| API | FastAPI + Uvicorn |
| Container | Docker, Docker Compose |
| Dataset | [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) |

## Cấu trúc thư mục

```
Recommendation-System/
├── data/
│   ├── raw/ml-latest-small/     # Dataset gốc (tải về khi chạy)
│   └── processed/               # ratings_clean.csv, user_item_matrix.csv
├── notebooks/figures/           # Biểu đồ EDA
├── src/
│   ├── data_preprocessing.py    # Tiền xử lý dữ liệu
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── evaluate.py              # Precision@K, Recall@K
│   ├── collaborative/
│   │   ├── user_based.py        # User-based Collaborative Filtering
│   │   └── item_based.py        # Item-based Collaborative Filtering
│   ├── content_based/
│   │   └── content_filter.py    # TF-IDF Content-based Filtering
│   ├── hybrid/
│   │   └── hybrid_recommender.py # Weighted Hybrid Model
│   └── api/
│       └── main.py              # FastAPI endpoints
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Cài đặt & Chạy

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Tải và xử lý dữ liệu

Tải [MovieLens ml-latest-small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip), giải nén vào `data/raw/`, sau đó:

```bash
python src/data_preprocessing.py
```

### 3. Chạy API

```bash
uvicorn src.api.main:app --reload
```

API chạy tại `http://localhost:8000`

### 4. Chạy bằng Docker

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/recommend/{user_id}?n=10` | Top-N gợi ý cho user |
| GET | `/similar/{movie_id}?n=10` | Top-N phim tương tự |
| GET | `/health` | Health check |

### Ví dụ

```bash
# Gợi ý cho user 1
curl http://localhost:8000/recommend/1?n=5

# Phim tương tự với movie 1
curl http://localhost:8000/similar/1?n=5
```

**Response mẫu:**
```json
{
  "user_id": 1,
  "recommendations": [
    { "movieId": 589, "title": "Terminator 2: Judgment Day (1991)", "score": 0.56 },
    { "movieId": 1200, "title": "Aliens (1986)", "score": 0.54 }
  ]
}
```

## Các mô hình

### User-based Collaborative Filtering
Tính Cosine Similarity giữa các users, tổng hợp điểm từ N users tương tự nhất để gợi ý phim chưa xem.

### Item-based Collaborative Filtering
Tính Cosine Similarity giữa các items, gợi ý phim tương tự với những phim user đã đánh giá cao.

### Content-based Filtering
Dùng TF-IDF trên genre của phim, gợi ý các phim có nội dung tương tự với lịch sử xem của user.

### Hybrid Recommender
Kết hợp cả 3 mô hình với trọng số:
- User-based CF: **40%**
- Item-based CF: **30%**
- Content-based: **30%**

## EDA Insights

- **610** users, **2269** movies, **81,116** ratings
- Sparsity: **94.14%**
- Top genres: Drama, Comedy, Action, Thriller, Adventure
