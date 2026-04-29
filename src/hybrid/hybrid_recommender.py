import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from collaborative.user_based import UserBasedCF, load_matrix
from collaborative.item_based import ItemBasedCF
from content_based.content_filter import ContentBasedFilter

PROCESSED_DIR = Path("data/processed")


class HybridRecommender:
    """Weighted hybrid: combines User-CF, Item-CF, and Content-based scores."""

    def __init__(self, weights: dict = None):
        self.weights = weights or {"user_cf": 0.4, "item_cf": 0.3, "content": 0.3}
        matrix = load_matrix()
        matrix.index = matrix.index.astype(int)
        matrix.columns = matrix.columns.astype(int)
        self.user_cf = UserBasedCF(matrix)
        self.item_cf = ItemBasedCF(matrix)
        self.content = ContentBasedFilter()
        self.ratings = pd.read_csv(PROCESSED_DIR / "ratings_clean.csv")
        self.movie_titles = self.content.movie_titles

    def _normalize(self, series: pd.Series) -> pd.Series:
        if series.max() == series.min():
            return series * 0
        return (series - series.min()) / (series.max() - series.min())

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        user_recs = self._normalize(self.user_cf.recommend(user_id, n=50))
        item_recs = self._normalize(self.item_cf.recommend(user_id, n=50))
        content_recs = self._normalize(
            self.content.recommend(user_id, self.ratings, n=50).set_index("movieId")["score"]
        )

        all_movies = set(user_recs.index) | set(item_recs.index) | set(content_recs.index)
        scores = {}
        for movie in all_movies:
            score = (
                self.weights["user_cf"] * user_recs.get(movie, 0)
                + self.weights["item_cf"] * item_recs.get(movie, 0)
                + self.weights["content"] * content_recs.get(movie, 0)
            )
            scores[movie] = score

        top_n = pd.Series(scores).sort_values(ascending=False).head(n)
        result = top_n.reset_index()
        result.columns = ["movieId", "score"]
        result["title"] = result["movieId"].map(self.movie_titles)
        return result


if __name__ == "__main__":
    model = HybridRecommender()
    sample_user = 1
    recs = model.recommend(sample_user)
    print(f"Top 10 hybrid recommendations for user {sample_user}:")
    print(recs[["title", "score"]].to_string(index=False))
