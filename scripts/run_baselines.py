import polars as pl
from pathlib import Path
import time
import sys

sys.path.append(".")

from src.models.baselines import GlobalMeanPredictor, UserMeanPredictor, MovieMeanPredictor, BiasBaselinePredictor
from src.evaluation.metrics import calculate_rmse, calculate_map_at_k

def main():
    print("Loading data lazily...")
    processed_path = Path("data/processed")
    df = pl.scan_parquet(processed_path / "interactions.parquet")
    
    print("Creating train/val split (80/20 on 10% sample to prevent OOM)...")
    # deterministic split using hash
    df = df.with_columns(
        (pl.col("customer_id").hash() + pl.col("movie_id").hash()).alias("hash_val")
    )
    
    # Filter to 10% of total data to fit in RAM
    df_sample = df.filter(pl.col("hash_val") % 100 < 10)
    
    # Split the 10% sample into 80/20 train/val
    train_df = df_sample.filter(pl.col("hash_val") % 10 < 8).collect()
    val_df = df_sample.filter(pl.col("hash_val") % 10 >= 8).collect()
    
    print(f"Train size: {train_df.height:,}")
    print(f"Val size: {val_df.height:,}")
    
    models = {
        "Global Mean": GlobalMeanPredictor(),
        "User Mean": UserMeanPredictor(),
        "Movie Mean": MovieMeanPredictor(),
        "Bias Baseline": BiasBaselinePredictor()
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(train_df)
        print(f"Training took {time.time()-t0:.2f}s")
        
        print(f"Predicting with {name}...")
        preds = model.predict(val_df)
        
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
