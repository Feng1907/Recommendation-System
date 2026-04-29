import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("notebooks/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROCESSED_DIR / "ratings_clean.csv")
df["genre_list"] = df["genres"].str.split("|")

print("=== Dataset Overview ===")
print(f"Users   : {df['userId'].nunique()}")
print(f"Movies  : {df['movieId'].nunique()}")
print(f"Ratings : {len(df)}")
print(f"Rating range : {df['rating'].min()} – {df['rating'].max()}")
sparsity = 1 - len(df) / (df['userId'].nunique() * df['movieId'].nunique())
print(f"Sparsity: {sparsity:.2%}")
print()

# 1. Rating distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["rating"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Rating Distribution")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Count")

# 2. Ratings per user
ratings_per_user = df.groupby("userId").size()
axes[1].hist(ratings_per_user, bins=30, color="coral", edgecolor="white")
axes[1].set_title("Ratings per User")
axes[1].set_xlabel("Number of Ratings")
axes[1].set_ylabel("Number of Users")
plt.tight_layout()
plt.savefig(FIG_DIR / "rating_distribution.png", dpi=120)
print("Saved: rating_distribution.png")

# 3. Top 20 most rated movies
fig, ax = plt.subplots(figsize=(10, 6))
top_movies = df.groupby("title").size().nlargest(20)
top_movies.plot(kind="barh", ax=ax, color="teal")
ax.set_title("Top 20 Most Rated Movies")
ax.set_xlabel("Number of Ratings")
plt.tight_layout()
plt.savefig(FIG_DIR / "top_movies.png", dpi=120)
print("Saved: top_movies.png")

# 4. Genre frequency
from collections import Counter
all_genres = [g for genres in df["genre_list"].dropna() for g in genres]
genre_counts = pd.Series(Counter(all_genres)).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
genre_counts.plot(kind="bar", ax=ax, color="mediumpurple")
ax.set_title("Genre Frequency")
ax.set_xlabel("Genre")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "genre_frequency.png", dpi=120)
print("Saved: genre_frequency.png")

print("\nTop 5 genres:", genre_counts.head().to_dict())
