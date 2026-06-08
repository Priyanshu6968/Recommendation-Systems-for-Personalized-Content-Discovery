from surprise import SVD, Dataset, Reader
import polars as pl
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit

class SurpriseSVDModel:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, verbose=False)
        self.reader = Reader(rating_scale=(1, 5))
        
    def _to_surprise_dataset(self, df: pl.DataFrame):
        pdf = df.select(["customer_id", "movie_id", "rating"]).to_pandas()
        return Dataset.load_from_df(pdf, self.reader)

    def fit(self, train_df: pl.DataFrame):
        data = self._to_surprise_dataset(train_df)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        
    def predict(self, test_df: pl.DataFrame):
        pdf = test_df.select(["customer_id", "movie_id"]).to_pandas()
        preds = []
        for uid, iid in zip(pdf["customer_id"], pdf["movie_id"]):
            pred = self.model.predict(uid, iid).est
            preds.append(pred)
        return preds

class ImplicitALSModel:
    def __init__(self, factors=100, iterations=15, regularization=0.05):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, 
            regularization=regularization, 
            iterations=iterations, 
            use_gpu=False
        )
        self.user_mapping = {}
        self.item_mapping = {}
        self.global_mean = 3.6
        
    def fit(self, train_df: pl.DataFrame):
        self.global_mean = train_df["rating"].mean()
        users = train_df["customer_id"].to_numpy()
        items = train_df["movie_id"].to_numpy()
        ratings = train_df["rating"].to_numpy()
        
        unique_users, u_indices = np.unique(users, return_inverse=True)
        unique_items, i_indices = np.unique(items, return_inverse=True)
        
        self.user_mapping = {u: idx for idx, u in enumerate(unique_users)}
        self.item_mapping = {i: idx for idx, i in enumerate(unique_items)}
        
        # implicit >= 0.5.0 expects user-item matrix: shape (users, items)
        user_item_data = csr_matrix((ratings, (u_indices, i_indices)), shape=(len(unique_users), len(unique_items)))
        
        self.model.fit(user_item_data)
        
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

    def predict(self, test_df: pl.DataFrame):
        users = test_df["customer_id"].to_numpy()
        items = test_df["movie_id"].to_numpy()
        
        preds = []
        for u, i in zip(users, items):
            u_idx = self.user_mapping.get(u, -1)
            i_idx = self.item_mapping.get(i, -1)
            
            if u_idx != -1 and i_idx != -1:
                # Raw dot product of factors
                pred = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
                
                # Implicit ALS optimizes for confidence/ranking, not explicit 1-5 RMSE.
                # To bring it roughly to scale, we might need to adjust, but let's see its raw power.
            else:
                pred = self.global_mean
            preds.append(pred)
            
        return np.clip(preds, 1.0, 5.0)
