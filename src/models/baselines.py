import polars as pl
import numpy as np

class GlobalMeanPredictor:
    def fit(self, df):
        self.global_mean = df["rating"].mean()
        
    def predict(self, df):
        return np.full(df.height, self.global_mean)

class UserMeanPredictor:
    def fit(self, df):
        self.global_mean = df["rating"].mean()
        self.user_means = df.group_by("customer_id").agg(pl.col("rating").mean().alias("user_mean"))
        
    def predict(self, df):
        pred_df = df.join(self.user_means, on="customer_id", how="left")
        return pred_df["user_mean"].fill_null(self.global_mean).to_numpy()

class MovieMeanPredictor:
    def fit(self, df):
        self.global_mean = df["rating"].mean()
        self.movie_means = df.group_by("movie_id").agg(pl.col("rating").mean().alias("movie_mean"))
        
    def predict(self, df):
        pred_df = df.join(self.movie_means, on="movie_id", how="left")
        return pred_df["movie_mean"].fill_null(self.global_mean).to_numpy()

class BiasBaselinePredictor:
    """ r_ui = mu + b_u + b_i """
    def fit(self, df, reg_u=10, reg_i=25):
        self.global_mean = df["rating"].mean()
        
        # Calculate movie bias: b_i = sum(r_ui - mu) / (count + reg_i)
        movie_bias = df.group_by("movie_id").agg(
            ((pl.col("rating") - self.global_mean).sum() / (pl.count() + reg_i)).alias("b_i")
        )
        self.movie_bias = movie_bias
        
        # Calculate user bias: b_u = sum(r_ui - mu - b_i) / (count + reg_u)
        df_with_bi = df.join(movie_bias, on="movie_id", how="left").with_columns(
            pl.col("b_i").fill_null(0.0)
        )
        
        user_bias = df_with_bi.group_by("customer_id").agg(
            ((pl.col("rating") - self.global_mean - pl.col("b_i")).sum() / (pl.count() + reg_u)).alias("b_u")
        )
        self.user_bias = user_bias

    def predict(self, df):
        pred_df = df.join(self.movie_bias, on="movie_id", how="left").join(self.user_bias, on="customer_id", how="left")
        
        pred = self.global_mean + pred_df["b_i"].fill_null(0.0) + pred_df["b_u"].fill_null(0.0)
        return pred.clip(1.0, 5.0).to_numpy()
