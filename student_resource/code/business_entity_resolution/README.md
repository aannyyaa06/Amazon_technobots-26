# Business Entity Resolution (Choice A)

Fuzzy/string features + TF-IDF + structural features → LightGBM.

No Sentence Transformers, embeddings, cross-encoders, or rerankers.

## Problem

Match Source 1 business records to zero/one/many records in Source 2 and Source 3.

TSVs must be read with `sep="\t"`. Country is an open-set string (train: US/India; test also has France).

## How to run (Phase 1)

From `code/business_entity_resolution/`:

```bash
pip install -r requirements.txt
python -m src.pipeline --phase 1
```

Phase 1 is read-only exploration. Stats are written to `output/exploration/`.
