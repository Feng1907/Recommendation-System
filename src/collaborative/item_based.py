import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def load_matrix() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "user_item_matrix.csv", index_col=0)


class ItemBasedCF:
    def __init__(self, matrix: pd.DataFrame):
        self.matrix = matrix
        self._fit()

    def _fit(self):
        filled = self.matrix.fillna(0).values.T  # items as rows
        sim = cosine_similarity(filled)
        self.sim_df = pd.DataFrame(sim, index=self.matrix.columns, columns=self.matrix.columns)

    def get_similar_items(self, movie_id: int, n: int = 10) -> pd.Series:
        return self.sim_df[movie_id].drop(movie_id).nlargest(n)

    def recommend(self, user_id: int, n: int = 10) -> pd.Series:
        user_ratings = self.matrix.loc[user_id].dropna()
        scores = {}
        for movie_id, rating in user_ratings.items():
            similar = self.sim_df[movie_id].drop(user_ratings.index, errors="ignore")
            for sim_movie, sim_score in similar.items():
                scores[sim_movie] = scores.get(sim_movie, 0) + sim_score * rating

        return pd.Series(scores).sort_values(ascending=False).head(n)


if __name__ == "__main__":
    matrix = load_matrix()
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)

    model = ItemBasedCF(matrix)
    sample_user = matrix.index[0]
    recs = model.recommend(sample_user)
    print(f"Top 10 item-based recommendations for user {sample_user}:")
    print(recs)
