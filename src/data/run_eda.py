import polars as pl
from pathlib import Path
import yaml

def load_config():
    with open("configs/default_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    processed_path = Path(config["data"]["processed_path"])
    
    print("Loading data...")
    user_features = pl.read_parquet(processed_path / "user_features.parquet")
    movie_features = pl.read_parquet(processed_path / "movie_features.parquet")
    
    # 1. Sparsity & Global Metrics
    n_users = user_features.height
    n_movies = movie_features.height
    n_ratings = user_features["user_rating_count"].sum()
    
    possible_ratings = n_users * n_movies
    sparsity = 1.0 - (n_ratings / possible_ratings)
    
    print("--- GLOBAL METRICS ---")
    print(f"Total Users: {n_users:,}")
    print(f"Total Movies: {n_movies:,}")
    print(f"Total Ratings: {n_ratings:,}")
    print(f"Matrix Sparsity: {sparsity:.4%}")
    print(f"Matrix Density: {1.0 - sparsity:.4%}")
    
    # 2. User Activity
    print("\n--- USER ACTIVITY ---")
    print(user_features["user_rating_count"].describe())
    
    # 3. Movie Popularity (The Long Tail)
    print("\n--- MOVIE POPULARITY ---")
    print(movie_features["movie_rating_count"].describe())
    
    # Top 5 most rated movies
    top_movies = movie_features.sort("movie_rating_count", descending=True).head(5)
    print("\nTop 5 Most Rated Movies:")
    print(top_movies.select(["title", "movie_rating_count", "movie_mean_rating"]))
    
    # 4. Rating Distribution approximations
    print("\n--- RATING AVERAGES ---")
    print(f"Global Average Rating (User-wise mean of means): {user_features['user_mean_rating'].mean():.2f}")
    
    print("\nEDA Calculations Complete.")

if __name__ == "__main__":
    main()
