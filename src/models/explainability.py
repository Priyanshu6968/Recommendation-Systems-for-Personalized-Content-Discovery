import pandas as pd

class ExplanationEngine:
    def __init__(self, cf_model, train_df, movie_metadata_df=None):
        """
        cf_model: An instance of SurpriseCFModel (Item-based)
        train_df: polars or pandas dataframe used for training (must have customer_id, movie_id, rating)
        movie_metadata_df: optional dataframe with movie_id and title to produce readable text.
        """
        self.model = cf_model.model
        self.train_df = train_df.to_pandas() if hasattr(train_df, "to_pandas") else train_df
        self.movie_metadata = {}
        if movie_metadata_df is not None:
            pdf = movie_metadata_df.to_pandas() if hasattr(movie_metadata_df, "to_pandas") else movie_metadata_df
            self.movie_metadata = dict(zip(pdf["movie_id"], pdf["title"]))
            
    def get_movie_name(self, movie_id):
        return self.movie_metadata.get(movie_id, f"Movie #{movie_id}")

    def explain(self, user_id, recommended_movie_id):
        try:
            inner_item_id = self.model.trainset.to_inner_iid(recommended_movie_id)
        except ValueError:
            return f"We recommend '{self.get_movie_name(recommended_movie_id)}' because it's broadly popular."

        # Get all items the user has rated in the training set
        try:
            inner_user_id = self.model.trainset.to_inner_uid(user_id)
        except ValueError:
            return f"We recommend '{self.get_movie_name(recommended_movie_id)}' based on general popularity (new user)."

        user_ratings = self.model.trainset.ur[inner_user_id]
        
        # Filter for items the user liked (rating >= 4)
        liked_items = [(iid, rating) for iid, rating in user_ratings if rating >= 4.0]
        
        if not liked_items:
            return f"We recommend '{self.get_movie_name(recommended_movie_id)}', though we don't have strong prior signals from your ratings."

        # Find the liked item most similar to the recommended item
        best_match_inner_id = None
        highest_sim = -1
        
        for iid, rating in liked_items:
            sim = self.model.sim[inner_item_id, iid]
            if sim > highest_sim:
                highest_sim = sim
                best_match_inner_id = iid
                
        if best_match_inner_id is not None and highest_sim > 0.1:
            raw_match_id = self.model.trainset.to_raw_iid(best_match_inner_id)
            match_name = self.get_movie_name(raw_match_id)
            rec_name = self.get_movie_name(recommended_movie_id)
            
            return f"We recommend '{rec_name}' because you previously enjoyed '{match_name}' (Similarity Score: {highest_sim:.2f})."
        else:
            rec_name = self.get_movie_name(recommended_movie_id)
            return f"We recommend '{rec_name}' based on broader community trends."
