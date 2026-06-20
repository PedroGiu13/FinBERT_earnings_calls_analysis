# Earnings Call NLP Processing Pipeline

## Description
A data engineering pipeline designed to ingest, process, and structure financial earnings transcripts for sentiment analysis via FinBERT. The system extracts transcripts from a Hugging Face repository, isolates the unscripted Q&A sessions, dynamically filters for executive responses to prevent analyst signal contamination, and applies sliding-window token chunking via LangChain to satisfy the strict 512-token input limit of BERT-based transformer architectures.

## Directory Structure
.
├── config/
│   └── config.yaml             # Pipeline configuration (equity universe, bounds, I/O paths)
├── data/
│   ├── raw/                    # Immutable cache of Hugging Face dataset shards
│   ├── processed/              # Intermediate state: Filtered universe transcripts
│   └── features/               # Tensor-ready state: Chunked, token-bounded executive text
├── src/
│   ├── data_loader.py          # Network ingestion, caching, and vectorized Pandas filtering
│   ├── text_processor.py       # Dynamic executive speaker resolution and NLTK/LangChain chunking
│   └── utils/
│       └── logger.py           # Standardized execution logging
└── main.py                     # Pipeline orchestrator and state materialization management
