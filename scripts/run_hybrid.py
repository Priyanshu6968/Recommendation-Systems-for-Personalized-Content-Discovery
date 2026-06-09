import polars as pl
from pathlib import Path
import time
import sys
import numpy as np

sys.path.append(".")

from src.models.collaborative_filtering import SurpriseCFModel
from src.models.matrix_factorization import SurpriseSVDModel
from src.models.hybrid import WeightedEnsemble
from src.evaluation.metrics import calculate_rmse, calculate_map_at_k

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
    
    print(f"Train size: {train_df.height:,}")
    print(f"Val size: {val_df.height:,}")
    
    models_to_ensemble = [
        SurpriseSVDModel(),
        SurpriseCFModel(user_based=False) # Item-based CF
    ]
    weights = [0.7, 0.3] # 70% SVD, 30% Item-CF
    
    ensemble = WeightedEnsemble(models=models_to_ensemble, weights=weights)
    
    print(f"\nTraining Hybrid Ensemble (SVD: 70%, Item-CF: 30%)...")
    t0 = time.time()
    ensemble.fit(train_df)
    print(f"Training took {time.time()-t0:.2f}s")
    
    print("Predicting with Hybrid Ensemble...")
    t1 = time.time()
    preds = ensemble.predict(val_df)
    print(f"Prediction took {time.time()-t1:.2f}s")
    
    val_preds_df = val_df.with_columns(pl.Series(name="prediction", values=preds))
    
    rmse = calculate_rmse(val_df["rating"].to_numpy(), preds)
    print(f"\nRMSE: {rmse:.4f}")
    
    map10 = calculate_map_at_k(val_preds_df)
    print(f"MAP@10: {map10:.4f}")
    
    print("\n--- FINAL RESULTS ---")
    print(f"Hybrid Ensemble | RMSE: {rmse:.4f} | MAP@10: {map10:.4f}")

if __name__ == "__main__":
    main()
