"""
Evaluate recommendation models using Precision@K and Recall@K.
Uses leave-one-out: hold out the last rated item per user as ground truth.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from collaborative.user_based import UserBasedCF, load_matrix
from collaborative.item_based import ItemBasedCF

PROCESSED_DIR = Path("data/processed")


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def evaluate_cf(model, test_set: dict, k: int = 10) -> dict:
    precisions, recalls = [], []
    for user_id, relevant_items in test_set.items():
        try:
            recs = model.recommend(user_id, n=k)
            rec_list = list(recs.index)
            precisions.append(precision_at_k(rec_list, relevant_items, k))
            recalls.append(recall_at_k(rec_list, relevant_items, k))
        except Exception:
            continue
    return {
        f"Precision@{k}": round(np.mean(precisions), 4),
        f"Recall@{k}": round(np.mean(recalls), 4),
    }


def build_test_set(df: pd.DataFrame, n_test_users: int = 100) -> dict:
    """Hold out last-rated movie per user as ground truth."""
    df_sorted = df.sort_values("timestamp")
    test_set = {}
    matrix = load_matrix()
    valid_users = set(matrix.index.astype(int))
    eligible = [u for u in df["userId"].unique() if u in valid_users]
    sample_users = pd.Series(eligible).head(n_test_users)
    for user in sample_users:
        user_df = df_sorted[df_sorted["userId"] == user]
        last_movie = user_df.iloc[-1]["movieId"]
        test_set[user] = {last_movie}
    return test_set


# NOTE: Leave-one-out on a 2000+ item catalog yields near-zero P@K / R@K — this
# is expected and documented in RecSys literature. Results show relative comparison.
if __name__ == "__main__":
    matrix = load_matrix()
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)

    df = pd.read_csv(PROCESSED_DIR / "ratings_clean.csv")
    test_set = build_test_set(df, n_test_users=50)

    for k in [10, 20, 50]:
        print(f"\n=== K = {k} ===")
        print("Evaluating User-based CF...")
        user_model = UserBasedCF(matrix)
        print(f"  {evaluate_cf(user_model, test_set, k=k)}")

        print("Evaluating Item-based CF...")
        item_model = ItemBasedCF(matrix)
        print(f"  {evaluate_cf(item_model, test_set, k=k)}")
