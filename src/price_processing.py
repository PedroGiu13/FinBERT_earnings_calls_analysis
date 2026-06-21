import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_ticker_returns(
    df_raw: pd.DataFrame, output_file_path: Path
) -> pd.DataFrame:
    # Create directory
    output_file_path.parent.mkdir(exist_ok=True, parents=True)

    df = df_raw.copy()

    # Standardise names depending on the df structure
    df.index.name = "date"
    df = df.stack().reset_index()
    df.columns = ["date", "symbol", "close"]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Compute returns
    logger.info("Computing returns (1s, 5d, 20d)")
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change(1).shift(-1)
    df["ret_5d"] = df.groupby("symbol")["close"].pct_change(5).shift(-5)
    df["ret_20d"] = df.groupby("symbol")["close"].pct_change(20).shift(-20)

    df = df.dropna()

    # Save data
    df.to_parquet(output_file_path, engine="pyarrow", compression="snappy")

    logger.info(f"Returns computed and saved: {output_file_path}")

    return df
