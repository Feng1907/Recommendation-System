import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw/ml-latest-small")


class ContentBasedFilter:
    def __init__(self):
        self.movies = pd.read_csv(RAW_DIR / "movies.csv")
        self.movies["content"] = self.movies["genres"].str.replace("|", " ", regex=False)
        self._fit()

    def _fit(self):
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.movies["content"])
        sim = cosine_similarity(tfidf_matrix)
        self.sim_df = pd.DataFrame(sim, index=self.movies["movieId"], columns=self.movies["movieId"])
        self.movie_titles = self.movies.set_index("movieId")["title"]

    def get_similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        scores = self.sim_df[movie_id].drop(movie_id).nlargest(n)
        result = scores.reset_index()
        result.columns = ["movieId", "similarity"]
        result["title"] = result["movieId"].map(self.movie_titles)
        return result

    def recommend(self, user_id: int, ratings_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        # Weight similarity by user's rating
        liked = user_ratings[user_ratings["rating"] >= 4.0]["movieId"].tolist()
        if not liked:
            liked = user_ratings["movieId"].tolist()

        watched = set(user_ratings["movieId"].tolist())
        scores = {}
        for movie_id in liked:
            if movie_id not in self.sim_df.columns:
                continue
            similar = self.sim_df[movie_id].drop(list(watched), errors="ignore")
            for sim_movie, sim_score in similar.items():
                scores[sim_movie] = scores.get(sim_movie, 0) + sim_score

        result = pd.Series(scores).sort_values(ascending=False).head(n)
        df = result.reset_index()
        df.columns = ["movieId", "score"]
        df["title"] = df["movieId"].map(self.movie_titles)
        return df


if __name__ == "__main__":
    model = ContentBasedFilter()
    ratings = pd.read_csv(PROCESSED_DIR / "ratings_clean.csv")

    sample_user = int(ratings["userId"].iloc[0])
    recs = model.recommend(sample_user, ratings)
    print(f"Top 10 content-based recommendations for user {sample_user}:")
    print(recs[["title", "score"]].to_string(index=False))
