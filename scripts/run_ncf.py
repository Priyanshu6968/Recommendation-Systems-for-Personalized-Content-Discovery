import polars as pl
from pathlib import Path
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(".")

from src.models.ncf import NCF, NetflixDataset
from src.evaluation.metrics import calculate_rmse, calculate_map_at_k

def main():
    print("Loading data eagerly...")
    processed_path = Path("data/processed")
    df = pl.read_parquet(processed_path / "interactions.parquet")
    
    # Same dense subset as CF/SVD to allow apples-to-apples comparison
    print("Creating dense subset (top 2000 users, top 500 movies)...")
    top_movies = df.group_by("movie_id").len().sort("len", descending=True).head(500)["movie_id"].to_list()
    top_users = df.group_by("customer_id").len().sort("len", descending=True).head(2000)["customer_id"].to_list()
    
    df_sample = df.filter(pl.col("movie_id").is_in(top_movies) & pl.col("customer_id").is_in(top_users))
    
    # Map IDs to contiguous integers for Embeddings
    users = df_sample["customer_id"].to_numpy()
    items = df_sample["movie_id"].to_numpy()
    ratings = df_sample["rating"].to_numpy()
    
    unique_users, u_indices = np.unique(users, return_inverse=True)
    unique_items, i_indices = np.unique(items, return_inverse=True)
    
    # Add mapped indices to dataframe
    df_sample = df_sample.with_columns(
        pl.Series(name="u_idx", values=u_indices),
        pl.Series(name="i_idx", values=i_indices)
    )
    
    # 80/20 train/val split using a random sample
    train_df = df_sample.sample(fraction=0.8, seed=42)
    val_df = df_sample.join(train_df, on=["customer_id", "movie_id"], how="anti")
    
    print(f"Train size: {train_df.height:,}")
    print(f"Val size: {val_df.height:,}")
    
    # Create PyTorch datasets
    train_dataset = NetflixDataset(train_df["u_idx"].to_numpy(), train_df["i_idx"].to_numpy(), train_df["rating"].to_numpy())
    val_dataset = NetflixDataset(val_df["u_idx"].to_numpy(), val_df["i_idx"].to_numpy(), val_df["rating"].to_numpy())
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = NCF(num_users=len(unique_users), num_items=len(unique_items), embedding_dim=32).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs = 5
    print(f"\nTraining NCF for {epochs} epochs...")
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            
            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f}")
        
    print(f"Training took {time.time()-t0:.2f}s")
    
    print("Predicting with NCF...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for u, i, r in val_loader:
            u, i = u.to(device), i.to(device)
            preds = model(u, i)
            all_preds.extend(preds.cpu().numpy())
            
    all_preds = np.clip(all_preds, 1.0, 5.0)
    
    val_preds_df = val_df.with_columns(pl.Series(name="prediction", values=all_preds))
    
    rmse = calculate_rmse(val_df["rating"].to_numpy(), all_preds)
    print(f"\nRMSE: {rmse:.4f}")
    
    map10 = calculate_map_at_k(val_preds_df)
    print(f"MAP@10: {map10:.4f}")
    
    print("\n--- FINAL RESULTS ---")
    print(f"NCF             | RMSE: {rmse:.4f} | MAP@10: {map10:.4f}")

if __name__ == "__main__":
    main()
