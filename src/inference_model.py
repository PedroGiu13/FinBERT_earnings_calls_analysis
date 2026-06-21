from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _setup_hardware() -> torch.device:
    """Dynamically allocate the most optimal available computational hardware"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Hardware allocated: CUDA ({torch.cuda.get_device_name(0)})")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Hardware allocated: APPLE MPS")

    else:
        device = torch.device("cpu")
        logger.info("Hardware allocated: CPU")

    return device


def _load_finbert_model(
    model_name: str,
) -> Tuple[BertTokenizer, BertForSequenceClassification]:
    """Initialise FinBERT's tokenizer and pre-trained classification head"""

    tokenizer = BertTokenizer.from_pretrained(model_name)
    finbert = BertForSequenceClassification.from_pretrained(model_name)

    return tokenizer, finbert


def execute_finbert_inference(
    df: pd.DataFrame, output_file_path: Path, model_name: str, batch_size: int
) -> pd.DataFrame:
    # Create File Dir
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # setup model
    device = _setup_hardware()
    tokenizer, finbert = _load_finbert_model(model_name)

    finbert.to(device)  # type: ignore
    finbert.eval()  # type: ignore

    text_list = df["text_chunks"].to_list()

    pos_prob, neu_prob, neg_prob = [], [], []

    # Inference loop
    logger.info(
        f"Initiating forward passes. Total chunks: {len(text_list)} | Batch size: {batch_size}"
    )

    for i in tqdm(range(0, len(text_list), batch_size), desc="FinBERT Inference"):
        batch_text = text_list[i : i + batch_size]

        inputs: Any = tokenizer(
            batch_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = finbert(**inputs)  # type: ignore
            logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

        # Class mapping
        neu_prob.extend(probs[:, 0])
        pos_prob.extend(probs[:, 1])
        neg_prob.extend(probs[:, 2])

    df["pos_prob"] = pos_prob
    df["neu_prob"] = neu_prob
    df["neg_prob"] = neg_prob

    # Estimation of alpha signals
    df["chunk_sentiment"] = df["pos_prob"] - df["neg_prob"]
    logger.info("Aggregating chunk vectors to transcript-level data")

    group_keys = ["date", "symbol", "year", "quarter"]

    df_signals = (
        df.groupby(group_keys)
        .agg(
            total_chunks=("text_chunks", "count"),
            net_sentiment=("chunk_sentiment", "mean"),
            avg_pos_sentiment=("pos_prob", "mean"),
            avg_neg_sentiment=("neg_prob", "mean"),
        )
        .reset_index()
    )

    df_signals.to_parquet(output_file_path, engine="pyarrow", compression="snappy")
    logger.info(f"Signals saved to {output_file_path}")

    return df_signals
