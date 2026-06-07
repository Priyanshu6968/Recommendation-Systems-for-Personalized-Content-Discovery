import os
import yaml
import polars as pl
from pathlib import Path

def load_config():
    with open("configs/default_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    interim_path = Path(config["data"]["interim_path"])
    processed_path = Path(config["data"]["processed_path"])
    raw_path = Path(config["data"]["raw_path"])
    
    processed_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading interim parquet files...")
    parquet_files = list(interim_path.glob("data_*.parquet"))
    if not parquet_files:
        print("No interim parquet files found. Run make_dataset.py first.")
        return
        
    df = pl.scan_parquet(interim_path / "data_*.parquet")
    
    print("Computing features...")
    # User features
    user_features = df.group_by("customer_id").agg(
        pl.len().alias("user_rating_count"),
        pl.col("rating").mean().cast(pl.Float32).alias("user_mean_rating")
    )
    
    # Movie features
    movie_features = df.group_by("movie_id").agg(
        pl.len().alias("movie_rating_count"),
        pl.col("rating").mean().cast(pl.Float32).alias("movie_mean_rating")
    )
    
    # Load movie metadata
    titles_path = raw_path / "movie_titles.csv"
    if titles_path.exists():
        print("Loading movie metadata...")
        movie_data = []
        with open(titles_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                parts = line.strip().split(",", 2)
                if len(parts) == 3:
                    m_id, year, title = parts
                    movie_data.append({
                        "movie_id": int(m_id) if m_id.isdigit() else None,
                        "release_year": int(year) if year.isdigit() else None,
                        "title": title
                    })
                
        movies_df = pl.DataFrame(movie_data).drop_nulls(subset=["movie_id"])
        
        movie_features = movie_features.join(movies_df.lazy(), on="movie_id", how="left")
    
    print("Saving processed data...")
    # Save base interactions
    df.sink_parquet(processed_path / "interactions.parquet")
    
    # Save features
    user_features.collect().write_parquet(processed_path / "user_features.parquet")
    movie_features.collect().write_parquet(processed_path / "movie_features.parquet")
    
    print("Feature engineering complete!")

if __name__ == "__main__":
    main()
