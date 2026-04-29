import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def load_matrix() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "user_item_matrix.csv", index_col=0)


class UserBasedCF:
    def __init__(self, matrix: pd.DataFrame, n_similar: int = 20):
        self.matrix = matrix
        self.n_similar = n_similar
        self._fit()

    def _fit(self):
        filled = self.matrix.fillna(0).values
        sim = cosine_similarity(filled)
        self.sim_df = pd.DataFrame(sim, index=self.matrix.index, columns=self.matrix.index)

    def get_similar_users(self, user_id: int) -> pd.Series:
        return self.sim_df[user_id].drop(user_id).nlargest(self.n_similar)

    def recommend(self, user_id: int, n: int = 10) -> pd.Series:
        similar_users = self.get_similar_users(user_id)
        watched = self.matrix.loc[user_id].dropna().index.tolist()

        scores = {}
        for sim_user, sim_score in similar_users.items():
            rated = self.matrix.loc[sim_user].dropna()
            for movie_id, rating in rated.items():
                if movie_id not in watched:
                    scores[movie_id] = scores.get(movie_id, 0) + sim_score * rating

        return pd.Series(scores).sort_values(ascending=False).head(n)


if __name__ == "__main__":
    matrix = load_matrix()
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)

    model = UserBasedCF(matrix)
    sample_user = matrix.index[0]
    recs = model.recommend(sample_user)
    print(f"Top 10 recommendations for user {sample_user}:")
    print(recs)
