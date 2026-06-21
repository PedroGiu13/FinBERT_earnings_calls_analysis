from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.data_loader import (
    fetch_ticker_data,
    fetch_transcript_data,
    filter_transcript_universe,
)
from src.inference_model import execute_finbert_inference
from src.text_processor import process_structured_content
from src.utils.logger import get_logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Initialise environment
    load_dotenv()
    logger = get_logger(__name__)
    config = load_config()

    # Extract config variables
    data_config = config.get("data_directories", {})
    raw_dir = data_config.get("raw_dir")
    processed_dir = data_config.get("processed_dir")
    features_dir = data_config.get("features_dir")
    signals_dir = data_config.get("signals_dir")

    ingestion_data = config.get("ingestion_data", {})
    repo_id = ingestion_data.get("transcripts_repo")
    ticker_list = ingestion_data.get("tickers")
    start_date = ingestion_data.get("start_date")
    end_date = ingestion_data.get("end_date")

    nlp_data = config.get("nlp_model", {})
    hf_model_id = nlp_data.get("hf_model_id", "yiyanghkust/finbert-tone")
    batch_size = nlp_data.get("batch_size", 16)

    # 1. Data Ingestion and Processing
    logger.info("Starting Data Ingestion & Processing pipeline")

    # ===== Transcripts =====
    # Setup processed file path
    transcript_processed_file_path = (
        Path(processed_dir) / "earnings_transcripts.parquet"
    )

    # Execute pipeline if file exists
    if transcript_processed_file_path.exists():
        logger.warning("File already processed. Bypassing raw data extraction")
        df_transcripts_clean = pd.read_parquet(
            transcript_processed_file_path,
            engine="pyarrow",  # type: ignore
        )

    else:
        logger.info("Fetching raw transcript earnings data")
        df_transcripts_raw = fetch_transcript_data(repo_id=repo_id, output_path=raw_dir)

        logger.info("Filtering transcripts")
        df_transcripts_clean = filter_transcript_universe(
            df=df_transcripts_raw,
            tickers=ticker_list,
            start_year=int(start_date.split("-")[0]),
            end_year=int(end_date.split("-")[0]),
            output_file_path=Path(transcript_processed_file_path),
        )

    print(df_transcripts_clean.head())

    # ===== Transcripts =====
    # Setup raw and processed file path
    raw_ticker_file_name = "yfinance/raw_ticker_prices.parquet"
    ticker_raw_file_path = Path(raw_dir) / raw_ticker_file_name

    if ticker_raw_file_path.exists():
        logger.info("Tickers already fetch. Bypassing yfinance extraction")
        df_tickers_raw = pd.read_parquet(ticker_raw_file_path, engine="pyarrow")  # type: ignore

    else:
        logger.info("Fetching daily ticker prices")
        df_tickers_raw = fetch_ticker_data(
            tickers=ticker_list,
            start_date=start_date,
            end_date=end_date,
            output_file_path=ticker_raw_file_path,
        )

    print(df_tickers_raw.head())

    # 2. Text Processing pipeline
    logger.info("Starting Q&A text processing")

    # Setup processed file path and check if it already exists
    features_file_path = (
        Path(features_dir) / "text_processed_earnings_transcripts.parquet"
    )

    if features_file_path.exists():
        logger.warning("File already processed. Bypassing text processing")
        feature_matrix = pd.read_parquet(features_file_path, engine="pyarrow")  # type: ignore

    else:
        logger.info("Processing executives text data")
        feature_matrix = process_structured_content(
            df=df_transcripts_clean, output_file_path=features_file_path
        )

    print(feature_matrix.head())

    # 3. Inference and alpha generation
    logger.info("Starting FinBERT Inference Engine")

    signals_file_path = Path(signals_dir) / "transcript_signals.parquet"

    if signals_file_path.exists():
        logger.warning("Transcript signals found. Bypassing neural network inference")
        df_signals = pd.read_parquet(signals_file_path, engine="pyarrow")  # type: ignore

    else:
        df_signals = execute_finbert_inference(
            df=feature_matrix,
            output_file_path=signals_file_path,
            model_name=hf_model_id,
            batch_size=batch_size,
        )

    print(df_signals.head())


if __name__ == "__main__":
    main()
