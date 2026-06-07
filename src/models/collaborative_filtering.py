from surprise import KNNWithMeans, Dataset, Reader
import polars as pl
import pandas as pd

class SurpriseCFModel:
    def __init__(self, user_based=True, sim_options=None):
        if sim_options is None:
            sim_options = {
                'name': 'pearson_baseline',
                'user_based': user_based,
                'min_support': 5
            }
        self.model = KNNWithMeans(k=40, min_k=1, sim_options=sim_options, verbose=True)
        self.reader = Reader(rating_scale=(1, 5))
        
    def _to_surprise_dataset(self, df: pl.DataFrame):
        # Surprise requires a pandas dataframe with columns: user, item, rating
        pdf = df.select(["customer_id", "movie_id", "rating"]).to_pandas()
        return Dataset.load_from_df(pdf, self.reader)

    def fit(self, train_df: pl.DataFrame):
        data = self._to_surprise_dataset(train_df)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        
    def predict(self, test_df: pl.DataFrame):
        # Surprise prediction is slow if looped in pure python.
        # We will loop over the test set tuples.
        pdf = test_df.select(["customer_id", "movie_id"]).to_pandas()
        preds = []
        for uid, iid in zip(pdf["customer_id"], pdf["movie_id"]):
            pred = self.model.predict(uid, iid).est
            preds.append(pred)
        return preds
