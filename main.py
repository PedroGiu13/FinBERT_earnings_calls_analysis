from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.data_loader import fetch_data, filter_universe
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

    transcripts_data = config.get("transcripts_data", {})
    repo_id = transcripts_data.get("dataset_repo")
    ticker_list = transcripts_data.get("tickers")
    start_year = transcripts_data.get("start_year")
    end_year = transcripts_data.get("end_year")

    nlp_data = config.get("nlp_model", {})
    hf_model_id = nlp_data.get("hf_model_id", "yiyanghkust/finbert-tone")
    batch_size = nlp_data.get("batch_size", 16)

    # 1. Data Ingestion and Processing
    logger.info("Starting Data Ingestion & Processing pipeline")

    # Setup processed file path and check if it already exists
    processed_file_path = Path(processed_dir) / "earnings_transcripts.parquet"

    # Execute pipeline if file exists
    if processed_file_path.exists():
        logger.warning("File already processed. Bypassing raw data extraction")
        df_clean = pd.read_parquet(processed_file_path, engine="pyarrow")  # type: ignore

    else:
        logger.info("Fetching Data")
        df = fetch_data(repo_id=repo_id, output_path=raw_dir)

        logger.info("Filtering data")
        df_clean = filter_universe(
            df=df,
            tickers=ticker_list,
            start_year=start_year,
            end_year=end_year,
            output_file_path=Path(processed_file_path),
        )

    print(df_clean.head())

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
            df=df_clean, output_file_path=features_file_path
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
