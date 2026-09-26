# Business Entity Resolution — Choice A

**Approach:** fuzzy/string features + TF-IDF + structural features → LightGBM.

**Not used:** Sentence Transformers, embeddings, cross-encoders, rerankers, or Choice B/C.

This package lives at `code/business_entity_resolution/` inside the challenge `student_resource/` folder. Data stays in `dataset/` (not copied). Outputs go to `output/`.

---

## 1. Problem

Three independent business-record sources (TSV, `sep="\t"`):

| Source | Role |
|---|---|
| Source 1 | Deduplicated reference. Predict matches for every S1 entity. |
| Source 2 | Candidate records (`S2-` IDs) |
| Source 3 | Candidate records (`S3-` IDs) |

For each Source 1 entity, find **all** matching records from Source 2/3 (zero, one, or many).

**Columns (sources):** `entity_id`, `business_name`, `business_address`, `country`  
**Ground truth:** `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs; empty = no match)

**Metric:** macro-averaged **F0.5** at the Source 1 entity level (precision-weighted). Empty prediction on a true singleton scores 1.0; any predicted match on a singleton scores 0.0.

**Hard constraints:** no Cartesian product; blocking before pairwise features; no external data/APIs; country is an open string (do not hard-code US/India — test includes France); original columns stay untouched.

---

## 2. Planned pipeline (Choice A)

```
NORMALIZATION
  → BLOCKING / CANDIDATE GENERATION
  → CANDIDATE FEATURES (fuzzy + TF-IDF + structural)
  → LightGBM  P(match)
  → THRESHOLD TUNING (validation only)
  → matching_results.tsv
```

`candidate_pairs.tsv` is the **final** candidate set fed to the model. Every ID in `matching_results.tsv` must appear there first.

### Implementation order

| Phase | Status | What it does |
|---:|---|---|
| 1  | **Done** | Data loading + exploration (read-only) |
| 2  | **Done** | Normalization caching (`name_norm`, `address_norm`, `country_norm` → Parquet) |
| 3  | **Done** | 6-strategy blocking → deduplicated + capped candidate pairs Parquet |
| 4  | **Done** | Blocking recall evaluation (gates on ≥ 80 % recovery) |
| 5  | **Done** | Training pair construction — positives + hard negatives, S1-level split |
| 6  | **Done** | 17 Choice A features: 15 string/TF-IDF + 2 structural → feature matrix Parquet |
| 7  | **Done** | LightGBM training with early stopping; feature importance report |
| 8  | **Done** | Entity-level F0.5 on validation set |
| 9  | **Done** | Grid-search threshold tuning against macro F0.5 |
| 10 | **Done** | Error analysis: FP/FN/singleton false-merge report |
| 11 | **Done** | Test inference → `candidate_pairs.tsv` + `matching_results.tsv` |
| 12 | **Done** | `utils/validate_submission.py` wired into pipeline |


---

## 3. Phase 1 — what was implemented

Phase 1 **does not mutate data**. It only reads TSVs with an explicit tab separator and writes stats.

### Code added

```
code/business_entity_resolution/
├── requirements.txt
├── README.md                          ← this file
└── src/
    ├── config.py                      paths, seed=42, column names, chunk size
    ├── pipeline.py                    python -m src.pipeline --phase 1
    └── data/
        ├── loader.py                  pd.read_csv(..., sep="\t") in chunks
        ├── validation.py              header / tab checks
        └── explore.py                 stats + report writer

notebooks/01_data_exploration.ipynb    same explorer from a notebook
```

### How loading works

- Always `sep="\t"` (addresses and ID lists contain commas).
- `dtype=str` so IDs are not parsed as numbers.
- Chunked reads (`TSV_CHUNKSIZE = 50_000`) so ~2.4 GB of TSVs fit on a laptop.
- Schema check: expected headers, tab vs comma, `S1-`/`S2-`/`S3-` prefixes.

### What exploration measured

For every train/test source file:

- rows, unique IDs, duplicate IDs, prefix mismatches
- nulls per column
- unique business names / addresses
- country distribution (open-set; no US/India filter)

For ground truth:

- S1 coverage vs `train_source1.tsv`
- zero-match vs multi-match counts
- match-count histogram
- S2 vs S3 match ID counts
- invalid prefixes / duplicate IDs inside a list

Also computed the size of a naïve S1 × (S2+S3) product (must never be materialized).

### How to re-run Phase 1

From `code/business_entity_resolution/`:

```bash
pip install -r requirements.txt
python -m src.pipeline --phase 1
```

Or open `notebooks/01_data_exploration.ipynb`.

Runtime on this machine: **152.2 seconds**. Schema issues: **none**.

---

## 4. Phase 1 results

### Train sources

| File | Rows | Unique IDs | Dup IDs | Null names | Null addresses | Countries |
|---|---:|---:|---:|---:|---:|---|
| `train_source1.tsv` | 2,206,821 | 2,206,821 | 0 | 0 | 0 | US 1,323,633 / India 883,188 |
| `train_source2.tsv` | 5,034,616 | 5,034,616 | 0 | 6 | 168,967 | US 3,016,817 / India 2,017,799 |
| `train_source3.tsv` | 5,285,603 | 5,285,603 | 0 | 18 | 175,916 | US 3,170,056 / India 2,115,547 |

- All ID prefixes match the file.
- Unique names &lt; row counts (repeated / shared names).
- Source 1 addresses are almost all unique (2,130,606 unique vs 2,206,821 rows).
- S2/S3 have many **missing addresses** — name blocking cannot be skipped.

### Ground truth (`train_ground_truth.tsv`)

| Statistic | Value |
|---|---:|
| Rows / unique S1 entities | 2,206,821 |
| Duplicate S1 rows | 0 |
| S1 IDs missing from GT | 0 |
| GT IDs missing from S1 | 0 |
| Zero matches (singletons) | **123,247** (5.6%) |
| At least one match | 2,083,574 |
| Total matched IDs | 7,638,365 (all unique) |
| From Source 2 | 3,693,619 |
| From Source 3 | 3,944,746 |
| Invalid match prefixes | 0 |
| Intra-list duplicate IDs | 0 |
| Mean matches per S1 | 3.46 |

**Match-count distribution**

| # matches | # S1 entities |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

**Cartesian product size (do not compute):**  
2,206,821 × (5,034,616 + 5,285,603) ≈ **22.77 trillion** pairs.

### Test sources

| File | Rows | Dup IDs | Null names | Null addresses | Countries |
|---|---:|---:|---:|---:|---|
| `test_source1.tsv` | 1,732,544 | 0 | 0 | 0 | India 809,986 / US 663,106 / **France 259,452** |
| `test_source2.tsv` | 4,887,273 | 0 | 49 | 129,408 | India 2,312,565 / US 1,871,330 / **France 703,378** |
| `test_source3.tsv` | 5,082,316 | 0 | 61 | 136,098 | India 2,405,000 / US 1,945,701 / **France 731,615** |

Test has **three** country labels. Final `matching_results.tsv` must contain **exactly 1,732,544** rows (one per test S1).

### Implications for later phases

1. Blocking is mandatory; a full join is impossible.
2. Country must be compared as a normalized string, not a US/India one-hot.
3. Missing S2/S3 addresses are common; name features and name blocks matter.
4. ~5.6% true singletons: over-predicting matches hurts F0.5 badly.
5. Typical true set is 2–5 IDs; do not merge aggressively.

---

## 5. Phase 1 output files

| Path | Contents |
|---|---|
| `output/exploration/phase1_report.txt` | Human-readable Phase 1 report |
| `output/exploration/phase1_stats.json` | Same numbers as JSON (reproducible dump) |

Final challenge files (`output/candidate_pairs.tsv`, `output/matching_results.tsv`) are **not** produced yet — they come after inference (Phase 11).

---

## 6. Output format (when later phases exist)

**`output/candidate_pairs.tsv`**

```
source1_entity_id<TAB>candidate_entity_ids
```

**`output/matching_results.tsv`**

```
source1_entity_id<TAB>matched_entity_ids
```

Rules: one row per test S1; S2/S3 IDs only; no duplicate S1 rows; no duplicate IDs in a list; empty list allowed; every matched ID must have appeared as a candidate.

Validate later with:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 7. Reproducibility

- Fixed seed in `src/config.py`: `SEED = 42` (used from training onward).
- Same command re-runs Phase 1 and overwrites the exploration report.
- Dataset files are never written to.

---

## 8. Next step

**Phase 2 — Normalization:** add `name_norm`, `address_norm`, `country_norm` as new columns; leave originals unchanged. Legal-suffix and address-abbreviation rules; country = lowercase + trim so France works without hard-coding.
