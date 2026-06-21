from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_ic(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman IC cross-sectionally within each period, then aggregate into IC mean / IC IR / t-stat across periods."""

    logger.info("Estimating Information Coefficient")
    target_returns = ["ret_1d", "ret_5d", "ret_20d"]

    ic_results = []

    for ret_col in target_returns:
        period_ics = []

        # Check individual calendar quarters separately
        for _, group in df.groupby("calendar_q"):
            valid_q = group[["net_sentiment", ret_col]].dropna()

            if len(valid_q) < 3 or valid_q["net_sentiment"].nunique() < 3:
                continue

            rho, _ = stats.spearmanr(valid_q["net_sentiment"], valid_q[ret_col])
            if not np.isnan(rho):  # type: ignore
                period_ics.append(rho)

        # Compute each q ic metrics
        period_ics = np.array(period_ics)
        n_periods = len(period_ics)

        if n_periods < 2:
            logger.warning(f"Not enough periods with valid IC for {ret_col}")
            ic_mean = ic_std = icir = np.nan

        else:
            ic_mean = period_ics.mean()
            ic_std = period_ics.std(ddof=1)
            icir = ic_mean / ic_std if ic_std > 0 else np.nan
            t_stat = ic_mean / (ic_std - np.sqrt(n_periods)) if ic_std > 0 else np.nan

        ic_results.append(
            {
                "time_horizon": ret_col,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "t_stat": t_stat,
                "n_periods": n_periods,
            }
        )

    return pd.DataFrame(ic_results)


def _quantile_spread_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket sentiment into quartiles within each period (cross-sectionally), then average forward returns per bucket across periods."""

    logger.info("Computing Quantile Spread")
    target_returns = ["ret_1d", "ret_5d", "ret_20d"]

    def _bucket(group: pd.DataFrame) -> pd.Series:
        try:
            return pd.qcut(group["net_sentiment"], q=4, labels=["q1", "q2", "q3", "q4"])

        except ValueError:
            return pd.Series(np.nan, index=group.index)

    df["sentiment_quantile"] = pd.qcut(
        df["net_sentiment"], q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop"
    )

    df = df.copy()
    df["sentiment_quantile"] = df.groupby("calendar_q", group_keys=False).apply(_bucket)

    per_period_avg = df.groupby(["calendar_q", "sentiment_quantile"], observed=False)[
        target_returns
    ].mean()

    df_quantiles = per_period_avg.groupby("sentiment_quantile", observed=False).mean()

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

    df_merged["calendar_q"] = df_merged["date"].dt.to_period("Q")

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

    df_merged_to_save = df_merged.copy()
    df_merged_to_save["calendar_q"] = df_merged_to_save["calendar_q"].astype(str)

    df_merged_to_save.to_parquet(
        alpha_factors_file_path, engine="pyarrow", compression="snappy"
    )
    df_ic.to_parquet(ic_file_path, engine="pyarrow", compression="snappy")
    df_quantile.to_parquet(quantile_file_path, engine="pyarrow", compression="snappy")
    logger.info("Files saved succesfully")

    return df_merged
