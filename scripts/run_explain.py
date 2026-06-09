import polars as pl
from pathlib import Path
import sys

sys.path.append(".")

from src.models.collaborative_filtering import SurpriseCFModel
from src.models.explainability import ExplanationEngine

def load_movie_titles():
    # Load movie titles from dataset/movie_titles.csv
    try:
        df = pl.read_csv("dataset/movie_titles.csv", has_header=False, encoding="latin1", null_values=["NULL"], truncate_ragged_lines=True)
        df.columns = ["movie_id", "year", "title"]
        return df
    except Exception as e:
        print(f"Could not load movie metadata: {e}")
        return None

def main():
    print("Loading data...")
    processed_path = Path("data/processed")
    df = pl.read_parquet(processed_path / "interactions.parquet")
    movies_df = load_movie_titles()
    
    print("Creating dense subset (top 2000 users, top 500 movies) for quick training...")
    top_movies = df.group_by("movie_id").len().sort("len", descending=True).head(500)["movie_id"].to_list()
    top_users = df.group_by("customer_id").len().sort("len", descending=True).head(2000)["customer_id"].to_list()
    
    train_df = df.filter(pl.col("movie_id").is_in(top_movies) & pl.col("customer_id").is_in(top_users))
    
    print("Training Item-Based CF Model...")
    # Item-based CF
    model = SurpriseCFModel(user_based=False)
    model.fit(train_df)
    
    print("Initializing Explanation Engine...")
    engine = ExplanationEngine(cf_model=model, train_df=train_df, movie_metadata_df=movies_df)
    
    # Pick a popular user from our dense set to demonstrate
    sample_user = top_users[5]
    
    print(f"\nGenerating explanations for User {sample_user}...")
    user_history = train_df.filter(pl.col("customer_id") == sample_user)["movie_id"].to_list()
    
    candidates = [m for m in top_movies if m not in user_history]
    
    preds = []
    for m in candidates:
        est = model.model.predict(sample_user, m).est
        preds.append((m, est))
        
    preds.sort(key=lambda x: x[1], reverse=True)
    top_3 = preds[:3]
    
    print("\n--- TOP 3 RECOMMENDATIONS AND EXPLANATIONS ---")
    for movie_id, score in top_3:
        explanation = engine.explain(user_id=sample_user, recommended_movie_id=movie_id)
        print(f"Prediction Score: {score:.2f} / 5.0")
        print(f"Explanation: {explanation}\n")

if __name__ == "__main__":
    main()
