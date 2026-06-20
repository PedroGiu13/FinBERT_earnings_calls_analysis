## Overview
This repository contains an automated Natural Language Processing (NLP) pipeline designed to extract quantitative alpha signals and risk indicators from SEC 10-K filings. The system utilizes FinBERT for domain-specific sentiment analysis, processing raw regulatory text into structured, high-frequency metrics suitable for integration into quantitative equity strategies and portfolio risk models.

The architecture is modular, separating data acquisition, text normalization, machine learning inference, and signal generation into distinct operational phases.

### Pipeline Architecture

* **Ingestion Engine:** Interfaces with the SEC EDGAR API to systematically retrieve 10-K filings based on target Central Index Keys (CIKs). Implements explicit rate-limiting and compliant user-agent declaration to adhere to SEC automated scraping protocols.
* **Text Preprocessing & Extraction:** Parses raw HTML and text filings to isolate high-value sections, specifically Item 7 (Management's Discussion and Analysis) and Item 1A (Risk Factors). Executes HTML stripping, boilerplate removal, and sentence-level tokenization optimized for FinBERT's maximum context window constraint (512 tokens).
* **FinBERT Inference Layer:** Executes batch inference utilizing pre-trained financial language models. Calculates discrete sentiment probabilities (positive, negative, neutral) at the sentence level across extracted regulatory sections. Includes dynamic padding and hardware acceleration routing.
* **Analytics & Signal Generation:** Aggregates sentence-level probabilities into document- and section-level sentiment metrics. Computes year-over-year semantic drift and structural variation indices to quantify shifts in corporate disclosure language.
* **Data Persistence:** Exports structured, time-stamped alpha factors and metadata (CIK, fiscal year, sentiment distributions, discrepancy scores) to a target database or flat file architecture for quantitative backtesting consumption.
