import polars as pl
import numpy as np

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_map_at_k(df_predictions, user_col="customer_id", pred_col="prediction", true_rating_col="rating", k=10, relevance_threshold=4):
    """Fast MAP@K calculation using polars."""
    df = df_predictions.select([user_col, pred_col, true_rating_col])
    
    # Determine relevance
    df = df.with_columns(
        (pl.col(true_rating_col) >= relevance_threshold).cast(pl.UInt32).alias("relevant")
    )
    
    # Total relevant items per user
    total_rel = df.group_by(user_col).agg(pl.col("relevant").sum().alias("total_relevant"))
    
    # Sort predictions descending
    df = df.sort([user_col, pred_col], descending=[False, True])
    
    # Get top K
    top_k = df.group_by(user_col).head(k)
    
    # Calculate precision at rank i
    top_k = top_k.with_columns(
        pl.int_range(1, pl.len() + 1).over(user_col).alias("rank")
    )
    
    top_k = top_k.with_columns(
        pl.col("relevant").cum_sum().over(user_col).alias("cum_relevant")
    )
    
    top_k = top_k.with_columns(
        ((pl.col("cum_relevant") / pl.col("rank")) * pl.col("relevant")).alias("p_at_k")
    )
    
    # Sum P@K for each user
    ap_df = top_k.group_by(user_col).agg(pl.col("p_at_k").sum().alias("sum_p_at_k"))
    
    # Join with total relevant
    ap_df = ap_df.join(total_rel, on=user_col)
    
    # Calculate AP
    ap_df = ap_df.with_columns(
        pl.min_horizontal("total_relevant", pl.lit(k)).alias("min_rel_k")
    ).with_columns(
        pl.when(pl.col("min_rel_k") > 0)
        .then(pl.col("sum_p_at_k") / pl.col("min_rel_k"))
        .otherwise(0.0)
        .alias("ap")
    )
    
    return ap_df["ap"].mean()
