from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hybrid.hybrid_recommender import HybridRecommender

app = FastAPI(title="Recommendation System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender: HybridRecommender = None


@app.on_event("startup")
def load_model():
    global recommender
    recommender = HybridRecommender()


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[dict]


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, n: int = 10):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        recs = recommender.recommend(user_id, n=n)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    items = recs[["movieId", "title", "score"]].to_dict(orient="records")
    return RecommendResponse(user_id=user_id, recommendations=items)


@app.get("/similar/{movie_id}")
def similar_movies(movie_id: int, n: int = 10):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = recommender.content.get_similar_movies(movie_id, n=n)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    return {"movie_id": movie_id, "similar": result.to_dict(orient="records")}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": recommender is not None}
