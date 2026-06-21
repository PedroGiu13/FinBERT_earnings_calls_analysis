from pathlib import Path

import pandas as pd
import scipy.stats as stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_ic(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman Information Coefficient"""

    logger.info("Estimating Information Coefficient")
    target_returns = ["ret_1d", "ret_5d", "ret_20d"]

    ic_results = []

    for ret_col in target_returns:
        rho, p_val = stats.spearmanr(
            df["net_sentiment"].to_numpy(), df[ret_col].to_numpy(), nan_policy="omit"
        )
        ic_results.append({"time_horizon": ret_col, "ic_spearman": rho, "p_val": p_val})

    return pd.DataFrame(ic_results)


def _quantile_spread_analysis(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing Quantile Spread")
    target_returns = ["ret_1d", "ret_5d", "ret_20d"]
    df["sentiment_quantile"] = pd.qcut(
        df["net_sentiment"], q=4, labels=["q1", "q2", "q3", "q4"]
    )

    df_quantiles = df.groupby("sentiment_quantile", observed=False)[
        target_returns
    ].mean()

    return df_quantiles


def evaluate_alpha_factors(
    df_signals: pd.DataFrame, df_returns: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    # Create Output directory
    output_path.mkdir(parents=True, exist_ok=True)
    alpha_factors_file_path = output_path / "alpha_factors.parquet"
    ic_file_path = output_path / "spearman_ic.parquet"
    quantile_file_path = output_path / "quantile_analysis.parquet"

    # Load datasets
    logger.info("Merging feature matrices")
    df_merged = pd.merge(df_signals, df_returns, on=["symbol", "date"], how="inner")

    if df_merged.empty:
        raise ValueError("Empty merged dataframe. Check column formatting")

    # Compute IC
    df_ic = _compute_ic(df_merged)
    print("\n--- Information Coefficient (IC) ---")
    print(df_ic.to_string(index=False))

    # Compute Quantile Spread
    df_quantile = _quantile_spread_analysis(df_merged)
    print("\n--- Mean Forward Returns by Sentiment Quartile ---")
    print(df_quantile)

    # Save Files
    logger.info("Saving evaluation files")
    df_merged.to_parquet(
        alpha_factors_file_path, engine="pyarrow", compression="snappy"
    )
    df_ic.to_parquet(ic_file_path, engine="pyarrow", compression="snappy")
    df_quantile.to_parquet(quantile_file_path, engine="pyarrow", compression="snappy")
    logger.info("Files saved succesfully")

    return df_merged
