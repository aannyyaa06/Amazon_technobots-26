# Business Entity Resolution — Choice A

Entity resolution across three independent business-record sources. For every
Source 1 entity, find all matching records in Source 2 and Source 3 (zero,
one, or many matches allowed).

This implementation follows **Choice A**: fuzzy/string features + TF-IDF
features + structural features → LightGBM. No embeddings, no cross-encoders,
no Sentence Transformers.

Current status: **Phase 1 (data loading & exploration) complete.**
Phases 2–12 (normalization → blocking → features → LightGBM → threshold
tuning → inference → validation) are not yet implemented.

---

## 1. Problem

Three sources of business records:

- **Source 1** — deduplicated reference source
- **Source 2**
- **Source 3**

Each record has: `entity_id`, `business_name`, `business_address`, `country`.

For every Source 1 entity, the task is to return all matching Source 2 /
Source 3 records. A Source 1 entity can have zero, one, or multiple matches.

Ground truth (`train_ground_truth.tsv`) maps:

```
source1_entity_id    matched_entity_ids
```

where `matched_entity_ids` is a comma-separated list of Source 2 and/or
Source 3 IDs.

**Evaluation metric:** macro-averaged F0.5 at the Source 1 entity level.
F0.5 weights precision higher than recall, so the pipeline is built to avoid
aggressive false merges — an empty prediction on a true zero-match entity
gets full credit, while a wrong prediction there is penalized.

---

## 2. Data format

All files are **TSV**, not CSV — read with `sep="\t"`.

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

> **Note:** the `dataset/` folder is not included in this repository (files
> are hundreds of MB each and exceed GitHub's size limits). Place the
> official challenge TSVs under `dataset/train/` and `dataset/test/` before
> running the pipeline.

Constraints respected throughout:

- No external data — no business databases, registries, geocoding APIs, or
  internet lookups of any kind.
- `country` is treated as an **open-set string**. Training data contains only
  US and India; test data also contains France. Nothing is hard-coded to a
  fixed country list.
- No full Cartesian product between sources — candidate generation
  (blocking) always precedes expensive pairwise computation.

---

## 3. Project structure

```
business-entity-resolution/
│
├── dataset/
│   ├── train/   (not included — see Data format)
│   └── test/    (not included — see Data format)
│
├── output/
│   ├── exploration/
│   │   ├── phase1_report.txt
│   │   └── phase1_stats.json
│   ├── candidate_pairs.tsv        (generated in later phases)
│   └── matching_results.tsv       (generated in later phases)
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py
│       │   ├── data/
│       │   │   ├── loader.py
│       │   │   ├── validation.py
│       │   │   └── explore.py
│       │   ├── preprocessing/          (Phase 2, not yet implemented)
│       │   ├── blocking/               (Phase 3, not yet implemented)
│       │   ├── features/               (Phase 6, not yet implemented)
│       │   ├── models/                 (Phase 7, not yet implemented)
│       │   ├── evaluation/             (Phase 8–9, not yet implemented)
│       │   └── pipeline.py
│       ├── requirements.txt
│       └── README.md
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
└── utils/
    └── validate_submission.py
```

---

## 4. What's implemented — Phase 1 (data loading & exploration)

Phase 1 is strictly **read-only**. No normalization, no blocking, no
modeling. The original TSV files are never modified.

**Code:**

| File | Purpose |
|---|---|
| `src/config.py` | Paths, `SEED = 42`, column names, chunk size (50,000 rows) |
| `src/data/loader.py` | Loads TSVs with `sep="\t"`, `dtype=str`, in chunks |
| `src/data/validation.py` | Checks tab-separated headers and expected columns |
| `src/data/explore.py` | Computes statistics and writes the exploration report |
| `src/pipeline.py` | Entry point (`--phase 1` runs exploration only) |
| `notebooks/01_data_exploration.ipynb` | Same exploration, notebook form |

**What Phase 1 measures**, for every source in both train and test:

- Row count, unique `entity_id` count, duplicate ID count
- ID prefix validity (`S1-`, `S2-`, `S3-`)
- Null counts in `business_name`, `business_address`, `country`
- Country distribution
- Unique business name / address counts

**For ground truth specifically:**

- Number of Source 1 entities, and how many have zero vs. at least one match
- Total matched IDs, split by Source 2 vs. Source 3
- Distribution of match counts per Source 1 entity
- Whether every train Source 1 ID appears in ground truth (and vice versa)

**Also computed:** the size of a full Source 1 × (Source 2 + Source 3)
Cartesian product, to make explicit why blocking is required rather than
just asserted.

### Run it

```bash
cd code/business_entity_resolution
python -m src.pipeline --phase 1
```

Runtime on the full dataset: ~152 seconds. No schema issues found.

### Output

| File | Contents |
|---|---|
| `output/exploration/phase1_report.txt` | Human-readable summary |
| `output/exploration/phase1_stats.json` | Same numbers as JSON (reproducible) |

### Key findings

**Train**

- Source 1: 2,206,821 rows, no duplicate IDs, no nulls, countries limited to US and India
- Source 2: 5,034,616 rows; 6 missing names; 168,967 missing addresses
- Source 3: 5,285,603 rows; 18 missing names; 175,916 missing addresses
- Ground truth: one row per Source 1 entity, full ID coverage in both directions
- 123,247 Source 1 entities (5.6%) have zero matches
- 2,083,574 Source 1 entities have at least one match
- 7,638,365 total true matched IDs (~3.69M to Source 2, ~3.94M to Source 3), all valid
- Average 3.46 matches per Source 1 entity; most between 2 and 5, up to 11 in some cases
- A full Cartesian product would be ~22.8 trillion pairs — confirms blocking is mandatory

**Test**

- Source 1: 1,732,544 entities — this is exactly how many rows `matching_results.tsv` must contain
- Source 2: 4,887,273 rows; Source 3: 5,082,316 rows
- Countries: India, US, **and France** (France does not appear in training data)
- Missing name/address rates follow the same pattern as training data

These findings directly shape the rest of the pipeline: block before comparing,
treat country as a fully open string (since France is unseen at train time),
handle missing addresses safely, and avoid over-predicting matches given the
precision-weighted F0.5 metric.

---

## 5. Output format (produced in later phases)

**`output/candidate_pairs.tsv`**

```
source1_entity_id    candidate_entity_ids
```

**`output/matching_results.tsv`**

```
source1_entity_id    matched_entity_ids
```

Rules:

- Tab-separated
- Exactly one row per test Source 1 entity
- No duplicate Source 1 rows
- No duplicate IDs within a list
- Only valid Source 2 / Source 3 IDs
- Empty list is valid (no match)
- Every ID in `matching_results.tsv` must have already appeared as a
  candidate in `candidate_pairs.tsv`

Validate with:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The script prints `PASS` or a numbered list of errors.

---

## 6. Reproducibility

- Fixed random seed: `SEED = 42` in `src/config.py` (used from training
  onward in later phases)
- Re-running `python -m src.pipeline --phase 1` overwrites the exploration
  report with identical results
- Dataset files are never written to at any phase

---

## 7. Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Data loading + exploration | ✅ Done |
| 2 | Normalization (`name_norm`, `address_norm`, `country_norm`) | Next |
| 3 | Blocking / candidate generation | Not started |
| 4 | Blocking recall evaluation | Not started |
| 5 | Training pair construction | Not started |
| 6 | Choice A feature engineering (string + TF-IDF + structural) | Not started |
| 7 | LightGBM baseline | Not started |
| 8 | F0.5 evaluation | Not started |
| 9 | Threshold tuning | Not started |
| 10 | Error analysis | Not started |
| 11 | Test inference | Not started |
| 12 | Submission validation | Not started |

**Phase 2 (next):** add `name_norm`, `address_norm`, `country_norm` as new
columns, leaving originals unchanged. Includes legal-suffix normalization
(Pvt/Private, Ltd/Limited, Corp/Corporation), common address abbreviations
(Road/Rd, Street/St, Avenue/Ave), and country normalization limited to
lowercasing and trimming — so France works automatically without any
hard-coded country list.

---

## 8. Setup

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Place the challenge TSV files under `dataset/train/` and `dataset/test/`,
then run:

```bash
cd code/business_entity_resolution
python -m src.pipeline --phase 1
```

---

## 9. Constraints followed throughout

- No external data, registries, geocoding, or entity-resolution APIs
- No Sentence Transformers, embeddings, or cross-encoders (Choice A only —
  reserved for a later, separate phase if pursued)
- No full Cartesian product between sources at any stage
- CPU/RAM only, no GPU-only operations
- Country treated as an open-set string, never hard-coded
