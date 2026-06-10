import polars as pl
from pathlib import Path
import time
import sys

sys.path.append(".")

from src.models.matrix_factorization import SurpriseSVDModel
from src.evaluation.metrics import calculate_rmse, calculate_map_at_k, calculate_ranking_metrics

def main():
    print("Loading data eagerly...")
    processed_path = Path("data/processed")
    df = pl.read_parquet(processed_path / "interactions.parquet")
    
    print("Creating dense subset (top 2000 users, top 500 movies)...")
    top_movies = df.group_by("movie_id").len().sort("len", descending=True).head(500)["movie_id"].to_list()
    top_users = df.group_by("customer_id").len().sort("len", descending=True).head(2000)["customer_id"].to_list()
    
    df_sample = df.filter(pl.col("movie_id").is_in(top_movies) & pl.col("customer_id").is_in(top_users))
    
    # 80/20 train/val split
    train_df = df_sample.sample(fraction=0.8, seed=42)
    val_df = df_sample.join(train_df, on=["customer_id", "movie_id"], how="anti")
    
    model = SurpriseSVDModel()
    
    print("Training Funk SVD benchmark model...")
    model.fit(train_df)
    
    print("Generating predictions...")
    preds = model.predict(val_df)
    
    val_preds_df = val_df.with_columns(pl.Series(name="prediction", values=preds))
    
    print("\nCalculating Phase 10 Full Evaluation Framework...")
    rmse = calculate_rmse(val_df["rating"].to_numpy(), preds)
    map10 = calculate_map_at_k(val_preds_df, k=10, relevance_threshold=4)
    ranking_metrics = calculate_ranking_metrics(val_preds_df, k=10, relevance_threshold=4)
    
    print("\n--- COMPREHENSIVE SCORECARD ---")
    print(f"RMSE (Error):           {rmse:.4f}")
    print(f"MAP@10 (Ranking):       {map10:.4f}")
    print(f"Precision@10 (Ranking): {ranking_metrics['Precision@K']:.4f}")
    print(f"Recall@10 (Ranking):    {ranking_metrics['Recall@K']:.4f}")
    print(f"NDCG@10 (Ranking):      {ranking_metrics['NDCG@K']:.4f}")
    print(f"Hit Rate (Quality):     {ranking_metrics['Hit Rate']:.4f}")
    print(f"Catalog Coverage:       {ranking_metrics['Coverage']:.4f}")

if __name__ == "__main__":
    main()
