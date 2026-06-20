import re
from pathlib import Path
from typing import Any, Dict, List

import nltk
import pandas as pd
from langchain_text_splitters import NLTKTextSplitter

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def _extract_executive_responses(structured_content: List[Dict[str, Any]]) -> str:
    """Dynamically identifies company executives and extracts their responses during the Q&A session, discarding prepared remarks and analyst questions."""

    qa_index = 0

    transition_patterns = re.compile(
        r"(open the call to questions|first question|questions from analysts|turn the call over to.*for questions)",
        re.IGNORECASE,
    )

    # Define where Q&A begins
    for i, turn in enumerate(structured_content):
        if transition_patterns.search(turn.get("text", "")):
            qa_index = i
            break

    # If no valid Q&A is found return empty string
    if qa_index == 0:
        return ""

    # Build set containing only executive names
    prepared_remarks = structured_content[:qa_index]
    executives = {
        turn.get("speaker")
        for turn in prepared_remarks
        if turn.get("speaker") != "Operator"
    }

    # Retrieve the speech of executives
    qa_session = structured_content[qa_index:]
    executive_responses = [
        turn.get("text") for turn in qa_session if turn.get("speaker") in executives
    ]

    return " ".join(executive_responses)  # type: ignore


def _chunk_text_langchain(text: str) -> List[str]:
    if not text:
        return []

    splitter = NLTKTextSplitter(chunk_size=1750, chunk_overlap=250)

    return splitter.split_text(text)


def process_structured_content(
    df: pd.DataFrame, output_file_path: Path
) -> pd.DataFrame:
    output_file_path.parent.mkdir(exist_ok=True, parents=True)

    # Process Q&A text and keep only those from companies executives
    df["executives_qa_text"] = df["structured_content"].apply(
        _extract_executive_responses
    )

    # Keep only necessary columns
    df_clean = df.drop(columns=["structured_content"]).copy()
    df_clean = df_clean[df_clean["executives_qa_text"] != ""]

    # Process text into chunks
    df_clean["text_chunks"] = df_clean["executives_qa_text"].apply(
        lambda x: _chunk_text_langchain(x)
    )

    df_expanded = df_clean.explode("text_chunks", ignore_index=True)
    df_expanded = df_expanded.drop(columns=["executives_qa_text"])
    df_expanded = df_expanded[df_expanded["text_chunks"] != ""]

    df_expanded.to_parquet(output_file_path, engine="pyarrow", compression="snappy")

    return df_expanded
