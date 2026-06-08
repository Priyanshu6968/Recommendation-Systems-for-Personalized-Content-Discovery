import polars as pl
from pathlib import Path
import time
import sys

sys.path.append(".")

from src.models.matrix_factorization import SurpriseSVDModel, ImplicitALSModel
from src.evaluation.metrics import calculate_rmse, calculate_map_at_k

def main():
    print("Loading data eagerly...")
    processed_path = Path("data/processed")
    df = pl.read_parquet(processed_path / "interactions.parquet")
    
    # Same dense subset as CF to allow apples-to-apples comparison
    print("Creating dense subset (top 2000 users, top 500 movies)...")
    top_movies = df.group_by("movie_id").len().sort("len", descending=True).head(500)["movie_id"].to_list()
    top_users = df.group_by("customer_id").len().sort("len", descending=True).head(2000)["customer_id"].to_list()
    
    df_sample = df.filter(pl.col("movie_id").is_in(top_movies) & pl.col("customer_id").is_in(top_users))
    
    # 80/20 train/val split
    train_df = df_sample.sample(fraction=0.8, seed=42)
    val_df = df_sample.join(train_df, on=["customer_id", "movie_id"], how="anti")
    
    print(f"Train size: {train_df.height:,}")
    print(f"Val size: {val_df.height:,}")
    
    models = {
        "Implicit ALS": ImplicitALSModel(),
        "Funk SVD": SurpriseSVDModel()
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(train_df)
        print(f"Training took {time.time()-t0:.2f}s")
        
        print(f"Predicting with {name}...")
        t1 = time.time()
        preds = model.predict(val_df)
        print(f"Prediction took {time.time()-t1:.2f}s")
        
        val_preds_df = val_df.with_columns(pl.Series(name="prediction", values=preds))
        
        rmse = calculate_rmse(val_df["rating"].to_numpy(), preds)
        print(f"RMSE: {rmse:.4f}")
        
        map10 = calculate_map_at_k(val_preds_df)
        print(f"MAP@10: {map10:.4f}")
        
        results.append({"Model": name, "RMSE": rmse, "MAP@10": map10})
        
    print("\n--- FINAL RESULTS ---")
    for r in results:
        print(f"{r['Model']:<15} | RMSE: {r['RMSE']:.4f} | MAP@10: {r['MAP@10']:.4f}")

if __name__ == "__main__":
    main()
