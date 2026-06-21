import os
from pathlib import Path
from typing import List, cast

import pandas as pd
import yfinance as yf
from datasets import load_dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_transcript_data(repo_id: str, output_path: str) -> pd.DataFrame:
    """
    Downloads or retrieves from local cache the Hugging Face dataset, optimizes memory by stripping unneeded columns, and returns a Pandas DataFrame.
    """

    # Check if output path exists
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    try:
        dataset = load_dataset(
            path=repo_id,
            cache_dir=str(output_dir),
            split="train",
            token=os.environ.get("HF_TOKEN"),
        ).remove_columns(["content"])

        return cast(pd.DataFrame, dataset.to_pandas())

    except Exception as e:
        logger.warning(f"Error extracting dataset from Hugging Face: {str(e)}")
        raise


def filter_transcript_universe(
    df: pd.DataFrame,
    tickers: List[str],
    start_year: int,
    end_year: int,
    output_file_path: Path,
) -> pd.DataFrame:
    """Applies vectorized filtering to the ingested DataFrame based on the  target tickers and chronological bounds defined in the configuration."""

    # Setup directory
    output_file_path.parent.mkdir(exist_ok=True, parents=True)

    mask = df["symbol"].isin(tickers) & df["year"].between(start_year, end_year)

    df_clean = df.loc[mask].copy()

    if len(df_clean) == 0:
        raise ValueError("Empty dataframe")

    df_clean.to_parquet(output_file_path, engine="pyarrow", compression="snappy")
    logger.info(f"File saved to {output_file_path}")

    return df_clean


def fetch_ticker_data(
    tickers: List[str], start_date: str, end_date: str, output_file_path: Path
) -> pd.DataFrame:
    """Function data retreives the daily adjusted closing price of a given ticker universe from yahoo finance.

    Args:
        tickers (list[str]): universe of tickers to fetch.
        start_date (str): start data in 'YYYY-MM-DD' format.
        end_date (str): end data in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: df with tickers as columns and date as index
    """

    output_file_path.parent.mkdir(exist_ok=True, parents=True)

    # Fetch Data
    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        # Check if response is empty
        if data is None or data.empty:
            logger.warning("Empty response from yfinance")
            raise

        # Get only the daily adjusted closing price
        df = data["Close"]

        # Fallback if only one ticker is fetched
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers[0])

        # Save data
        df.to_parquet(output_file_path, engine="pyarrow", compression="snappy")
        logger.info(f"Ticker data succesfully saved: {output_file_path}")

    except Exception as e:
        logger.error(f"Failed to fetch yfinance data: {e}")
        raise

    return df
