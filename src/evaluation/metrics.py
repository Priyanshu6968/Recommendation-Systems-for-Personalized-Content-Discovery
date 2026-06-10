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

def calculate_ranking_metrics(df_predictions, user_col="customer_id", item_col="movie_id", pred_col="prediction", true_rating_col="rating", k=10, relevance_threshold=4):
    """
    Computes Precision@K, Recall@K, NDCG@K, Hit Rate, and Coverage.
    """
    df = df_predictions.select([user_col, item_col, pred_col, true_rating_col])
    
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
    
    # Coverage: unique items in top K across all users
    coverage = top_k[item_col].n_unique() / df[item_col].n_unique()
    
    # Calculate rank
    top_k = top_k.with_columns(
        pl.int_range(1, pl.len() + 1).over(user_col).alias("rank")
    )
    
    # Metrics aggregations per user
    user_metrics = top_k.group_by(user_col).agg([
        pl.col("relevant").sum().alias("hits_at_k"),
        # DCG@K: sum( (2^rel - 1) / log2(rank + 1) )
        # Since rel is binary 0/1 here, 2^rel - 1 is just rel.
        (pl.col("relevant") / np.log2(pl.col("rank") + 1)).sum().alias("dcg_at_k")
    ])
    
    # Join with total relevant
    user_metrics = user_metrics.join(total_rel, on=user_col)
    
    # IDCG calculation (Ideal DCG)
    # The max possible DCG is when all relevant items are at the top ranks
    def ideal_dcg(num_relevant, k):
        n = min(num_relevant, k)
        return float(sum(1.0 / np.log2(i + 1) for i in range(1, n + 1)))
        
    user_metrics = user_metrics.with_columns([
        (pl.col("hits_at_k") / k).alias("precision_at_k"),
        pl.when(pl.col("total_relevant") > 0).then(pl.col("hits_at_k") / pl.col("total_relevant")).otherwise(0.0).alias("recall_at_k"),
        (pl.col("hits_at_k") > 0).cast(pl.Float64).alias("hit_rate"),
        pl.col("total_relevant").map_elements(lambda r: ideal_dcg(r, k), return_dtype=pl.Float64).alias("idcg_at_k")
    ])
    
    # Calculate NDCG
    user_metrics = user_metrics.with_columns(
        pl.when(pl.col("idcg_at_k") > 0).then(pl.col("dcg_at_k") / pl.col("idcg_at_k")).otherwise(0.0).alias("ndcg_at_k")
    )
    
    return {
        "Precision@K": user_metrics["precision_at_k"].mean(),
        "Recall@K": user_metrics["recall_at_k"].mean(),
        "NDCG@K": user_metrics["ndcg_at_k"].mean(),
        "Hit Rate": user_metrics["hit_rate"].mean(),
        "Coverage": coverage
    }
