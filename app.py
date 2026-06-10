import streamlit as st
import polars as pl
from pathlib import Path
import sys

sys.path.append(".")

from src.models.collaborative_filtering import SurpriseCFModel
from src.models.explainability import ExplanationEngine

# Page config
st.set_page_config(page_title="Netflix RecSys Dashboard", page_icon="🎬", layout="wide")

@st.cache_resource(show_spinner="Loading data and training model... This takes ~15 seconds.")
def load_and_train():
    # Load movie metadata
    try:
        movies_df = pl.read_csv("dataset/movie_titles.csv", has_header=False, encoding="latin1", null_values=["NULL"], truncate_ragged_lines=True)
        movies_df.columns = ["movie_id", "year", "title"]
    except Exception as e:
        st.error(f"Failed to load movie metadata: {e}")
        movies_df = None

    # Load interactions
    processed_path = Path("data/processed")
    df = pl.read_parquet(processed_path / "interactions.parquet")
    
    # Dense subset
    top_movies = df.group_by("movie_id").len().sort("len", descending=True).head(500)["movie_id"].to_list()
    top_users = df.group_by("customer_id").len().sort("len", descending=True).head(2000)["customer_id"].to_list()
    
    train_df = df.filter(pl.col("movie_id").is_in(top_movies) & pl.col("customer_id").is_in(top_users))
    
    # Train model
    model = SurpriseCFModel(user_based=False)
    model.fit(train_df)
    
    # Initialize explainer
    engine = ExplanationEngine(cf_model=model, train_df=train_df, movie_metadata_df=movies_df)
    
    return train_df, movies_df, model, engine, top_users, top_movies

def main():
    st.title("🎬 Netflix Recommendation Dashboard")
    st.markdown("An interactive interface demonstrating our Item-Based Collaborative Filtering model with Explainable AI.")
    
    train_df, movies_df, model, engine, top_users, top_movies = load_and_train()
    
    if movies_df is not None:
        movies_dict = dict(zip(movies_df["movie_id"].to_list(), movies_df["title"].to_list()))
    else:
        movies_dict = {}
        
    def get_movie_name(movie_id):
        return movies_dict.get(movie_id, f"Movie #{movie_id}")

    # Sidebar
    st.sidebar.header("User Controls")
    selected_user = st.sidebar.selectbox("Select a Customer ID", top_users)
    num_recs = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
    
    # Split layout
    col1, col2 = st.columns([1, 1.5])
    
    # Display User History
    with col1:
        st.subheader("📜 User Watch History")
        user_history_df = train_df.filter(pl.col("customer_id") == selected_user)
        user_history_ids = user_history_df["movie_id"].to_list()
        
        display_history = user_history_df.with_columns(
            pl.col("movie_id").map_elements(lambda x: get_movie_name(x), return_dtype=pl.String).alias("Movie Title")
        ).select(["Movie Title", "rating"]).sort("rating", descending=True).to_pandas()
        
        st.dataframe(display_history, use_container_width=True, hide_index=True)

    # Display Recommendations
    with col2:
        st.subheader(f"✨ Top {num_recs} Recommendations")
        
        candidates = [m for m in top_movies if m not in user_history_ids]
        
        with st.spinner("Generating predictions..."):
            preds = []
            for m in candidates:
                est = model.model.predict(selected_user, m).est
                preds.append((m, est))
                
            preds.sort(key=lambda x: x[1], reverse=True)
            top_preds = preds[:num_recs]
            
        for i, (movie_id, score) in enumerate(top_preds):
            movie_title = get_movie_name(movie_id)
            explanation = engine.explain(selected_user, movie_id)
            
            with st.expander(f"**#{i+1}: {movie_title}** (Pred: {score:.2f} ⭐)", expanded=(i==0)):
                st.markdown(f"**Why this movie?**")
                st.info(explanation)

if __name__ == "__main__":
    main()
