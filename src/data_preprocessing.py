import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/ml-latest-small")
PROCESSED_DIR = Path("data/processed")


def load_raw():
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    tags = pd.read_csv(RAW_DIR / "tags.csv")
    return ratings, movies, tags


def preprocess(ratings: pd.DataFrame, movies: pd.DataFrame):
    # Drop duplicates
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"])

    # Filter users with at least 20 ratings and movies with at least 10 ratings
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()
    active_users = user_counts[user_counts >= 20].index
    popular_movies = movie_counts[movie_counts >= 10].index
    ratings = ratings[ratings["userId"].isin(active_users) & ratings["movieId"].isin(popular_movies)]

    # Merge with movie metadata
    df = ratings.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")

    # Normalize ratings to [0, 1]
    df["rating_norm"] = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min())

    # Expand genres into list
    df["genre_list"] = df["genres"].str.split("|")

    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df.pivot_table(index="userId", columns="movieId", values="rating")
    return matrix


def save(df: pd.DataFrame, matrix: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "ratings_clean.csv", index=False)
    matrix.to_csv(PROCESSED_DIR / "user_item_matrix.csv")
    print(f"Saved {len(df)} ratings, matrix shape: {matrix.shape}")


if __name__ == "__main__":
    ratings, movies, tags = load_raw()
    df = preprocess(ratings, movies)
    matrix = build_user_item_matrix(df)
    save(df, matrix)
