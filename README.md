# FinBERT Earnings Calls Sentiment — Alpha Signal Research

A production-structured NLP pipeline that applies **FinBERT** to earnings call transcripts to generate cross-sectional sentiment alpha signals, evaluated against short- and medium-term equity returns.

***

## Motivation

Earnings calls are one of the richest sources of forward-looking management guidance. This project tests whether the tone of executive language contains statistically meaningful information about subsequent stock returns.

***

## Pipeline Overview

```text
Earnings Transcripts (HuggingFace)          yfinance (Equity Prices)
          │                                           │
          ▼                                           ▼
   Data Ingestion &                         Closing Price Fetch
   Universe Filtering                       & Return Computation
          │                                 (1d / 5d / 20d fwd)
          ▼                                           │
   Text Processing                                    │
   (chunk & clean Q&A text)                           │
          │                                           │
          ▼                                           │
  FinBERT Inference                                   │
  (pos / neu / neg probs)                             │
          │                                           │
          ▼                                           │
   Signal Aggregation ──────────────────────────────▶ │
   (net_sentiment = avg pos − avg neg per transcript)  │
                                                       ▼
                                          Alpha Evaluation
                                     (Spearman IC, IC-IR, t-stat,
                                      Quantile Spread Analysis)
```

***

## Project Structure

```
├── config/
│   └── config.yaml              # Tickers, date range, model & directory config
├── src/
│   ├── data_loader.py           # Transcript & ticker ingestion
│   ├── text_processor.py        # Text chunking and cleaning
│   ├── inference_model.py       # FinBERT batched inference (CUDA / MPS / CPU)
│   ├── price_processing.py      # Return computation (1d, 5d, 20d)
│   ├── evaluation.py            # Spearman IC and quantile spread evaluation
│   └── utils/logger.py          # Structured logging
├── main.py                      # End-to-end pipeline entrypoint
└── requirements.txt
```

***

## Methodology

**Signal Construction:** Each earnings call transcript is split into text chunks from the executive Q&A section. FinBERT (`yiyanghkust/finbert-tone`) runs batched inference to produce positive/negative/neutral probabilities per chunk. The transcript-level alpha signal is:

```
net_sentiment = mean(pos_prob) − mean(neg_prob)
```

**Evaluation:**
- **Spearman IC** — cross-sectional rank correlation between `net_sentiment` and forward returns, computed per calendar quarter then aggregated into IC Mean, IC Std, IC-IR, and t-statistic.
- **Quantile Spread** — stocks bucketed into sentiment quartiles (Q1–Q4) within each period; mean forward returns per quartile averaged across all periods.

Forward return horizons tested: **1-day, 5-day, 20-day**.

***

## Results
### Spearman IC — Full vs. Filtered Sample

| Horizon | IC Mean (Full) | IC Mean (Filtered) | t-stat (Full) | t-stat (Filtered) | Periods (Full) | Periods (Filtered) |
|---------|---------------:|-------------------:|--------------:|------------------:|---------------:|-------------------:|
| 1-day   | 0.0237         | 0.0106             | 1.04          | 0.44              | 68             | 58                 |
| 5-day   | 0.0090         | 0.0019             | 0.43          | 0.09              | 68             | 58                 |
| 20-day  | −0.0010        | 0.0011             | −0.05         | 0.06              | 68             | 58                 |

### Interpretation

Across the full sample (68 quarters, 2008–2024), the sentiment signal shows a weak positive
relationship with near-term returns that does not reach statistical significance at conventional
thresholds (best case t = 1.04 at the 1-day horizon).

Excluding crisis periods (2008–09, H1 2020 — 13% of observations), the already-weak 1-day effect
shrinks by more than half (t = 0.44) and the 5-day effect is effectively eliminated (t = 0.09).
This indicates the modest signal observed in the full sample is concentrated in high-volatility
market regimes rather than reflecting a general relationship between earnings-call sentiment and
forward returns.

> **Conclusion:** FinBERT sentiment does not constitute a standalone tradeable factor at conventional significance levels in either normal or stressed market conditions.

## Discussion

The absence of a significant standalone alpha signal is itself an informative result. Generating robust, persistent alpha from publicly available information is genuinely hard — markets are competitive, and any signal derived from data accessible to all participants faces rapid arbitrage. The weak and regime-dependent IC observed here is consistent with the broader academic literature on NLP-based sentiment factors, where marginal predictability tends to erode once transaction costs and data availability are accounted for.

What this project demonstrates is the **end-to-end process** by which a systematic signal research pipeline is built and evaluated at an institutional standard: raw alternative data ingestion, domain-specific NLP inference, rigorous cross-sectional IC testing with regime robustness checks, and honest reporting of null results. This workflow — not the signal strength alone — is how quantitative alpha research is conducted in practice.

Potential directions to improve signal quality include sector-neutralised IC decomposition, combining sentiment with orthogonal factors such as earnings surprise (SUE) or price momentum, and fine-tuning FinBERT on proprietary or higher-quality transcript data.

***

## Setup

```bash
git clone https://github.com/PedroGiu13/FinBERT_earnings_calls_analysis.git
cd FinBERT_earnings_calls_analysis

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # add your HuggingFace token if needed
```

Edit `config/config.yaml` to set your ticker universe and date range, then run:

```bash
python main.py
```

The pipeline is **idempotent** — already-processed files are detected automatically and skipped,
so re-runs only execute missing steps.

***

## Tech Stack

- **NLP:** `transformers` (FinBERT), `torch` — with auto hardware routing (CUDA → MPS → CPU)
- **Data:** HuggingFace Datasets (transcripts), `yfinance` (prices)
- **Evaluation:** `scipy.stats` Spearman correlation, custom IC/ICIR framework
- **Storage:** Apache Parquet via `pyarrow` (Snappy compression)
- **Code Quality:** `pre-commit` hooks (ruff, black, isort)

***
