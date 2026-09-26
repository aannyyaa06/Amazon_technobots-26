# Business Entity Resolution Engine (Amazon ML Challenge 2026)

## Overview
High-performance Business Entity Resolution pipeline targeting maximum Macro $F_{0.5}$ score under competition constraints:
- Matches Source 1 entities against Source 2 and Source 3 records.
- Precision-heavy metric optimization: $F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$.
- Multi-branch candidate retrieval with high recall.
- Gradient Boosted Decision Tree (LightGBM) pairwise ranking with calibrated decision thresholds.

## Project Structure
```
code/business_entity_resolution/
├── src/
│   ├── blocking/               # High-recall multi-key candidate blocking engines
│   │   ├── candidate_generator.py
│   │   ├── name_blocking.py
│   │   └── address_blocking.py
│   ├── preprocessing/          # Normalization, legal entity stripping, postal code parser
│   │   ├── normalize.py
│   │   ├── normalize_names.py
│   │   └── normalize_addresses.py
│   ├── features/               # High-precision pairwise feature engineering
│   │   ├── feature_builder.py
│   │   ├── fast_features.py
│   │   ├── string_features.py
│   │   └── tfidf_features.py
│   ├── models/                 # LightGBM classifier & inference engines
│   │   ├── lightgbm_model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── evaluation/             # Official Macro F0.5 metrics & threshold tuning
│   │   ├── metrics.py
│   │   ├── threshold_tuning.py
│   │   └── validation.py
│   ├── run_finetuned_pipeline.py # End-to-end streaming test inference pipeline
│   ├── eval_ultra_precision.py   # Validation and feature tuning
│   └── config.py                 # File paths and global hyperparameters
├── requirements.txt
└── README.md
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run test set inference to reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
   ```bash
   python src/run_finetuned_pipeline.py
   ```
3. Validate output compliance with competition standards:
   ```bash
   python student_resource/utils/validate_submission.py
   ```
