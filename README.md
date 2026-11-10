# GeoQueryCA

Multilingual, context-aware benchmark dataset for geographic disambiguation, geotagging and retrieval: **2,001 queries, 82 languages, 125 countries and territories, 1,904 unique landmarks**.

**Current version: v1.1** — the version used for every result in the paper. All changes from v1.0 are listed record by record in `Dataset_Change_Log.xlsx`.

## Dataset

File: `GEOQUERYCA_DATASET_v1.1.csv` (UTF-8 with BOM, comma-separated, 2,001 rows, 13 columns)
SHA-256: `3fe343de307bba19107a42d4aafea6e143acf49d88fda864194647f6aea8a4cf`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Landmark` | string | Name of the landmark / geographic entity (gold label) | "Taj Mahal" |
| `City` | string | City or locality of the landmark | "Agra" |
| `Latitude` | float | Latitude, decimal degrees | 27.1751 |
| `Longitude` | float | Longitude, decimal degrees | 78.0421 |
| `Country` | string | Country or territory (ISO 3166 short name) | "India" |
| `Category` | string | One of 12 landmark categories | "Historical Site", "Religious Site", "Natural Attraction" |
| `Contextual_Query` | string | Natural-language query that describes the landmark without naming it | "श्वेत संगमरमर का मकबरा यमुना नदी के किनारे" |
| `Query_Language` | string | Language of the query (English name) | "Hindi", "English", "Telugu" |
| `Disambiguation_Cues` | string | Physical, architectural, cultural or geographic clues | "white marble mausoleum, Mughal architecture, Yamuna riverbank" |
| `Confidence_Score` | float | Composite reliability score in [0, 1] (paper Eq. 1) | 0.92 |
| `Region` | string | Macro-region | "Asia", "Europe", "Americas", "Africa", "Oceania" |
| `Nearby_Attractions` | string | Nearby landmarks and attractions | "Agra Fort, Mehtab Bagh" |
| `Cultural_Significance` | string | Historical, cultural and social context (retrieval passage) | "Mausoleum built by Mughal emperor Shah Jahan..." |

## Statistics (v1.1)
- Records: 2,001 query–landmark pairs; 1,904 unique landmarks; 1,148 cities/regions
- Languages: 82 (English 611; Indian vernaculars 564)
- Countries and territories: 125 (India 30.6% of records)
- Categories: 12
- Confidence score: mean 0.91 (σ = 0.073), range 0.67–1.00
- Coordinate validation (Wikidata/GeoNames, with country check): 1,450 of 1,904 landmarks matched; median distance 0.93 km

## Repository contents

| File | Purpose | Paper |
|---|---|---|
| `GEOQUERYCA_DATASET_v1.1.csv` | The dataset | — |
| `geoqueryca_pipeline_github_release.py` | Dataset profiling, zero-shot benchmark of six models, bootstrap confidence intervals, Holm–Bonferroni-corrected significance tests, mBERT fine-tuning on the landmark-grouped split (1,408 train / 590 test) | §4, §6.1–6.4, Tables 3–7, Figs. 3–5 |
| `geoqueryca_field_ablation.py` | Field ablation with mE5-large (six input configurations) | §6.5, Table 8 |
| `coord_validation_github_release.py` | Coordinate verification against Wikidata (P625) and GeoNames with an ISO 3166 country check; summary statistics and Figure 2 | §3.2, Table 2, Fig. 2 |
| `confidence_score_github_release.py` | Recomputes the Confidence_Score column; reproduces all 2,001 released values | §4.3, Eq. 1 |
| `annotation_validation.py` | Automated consistency checks, correction count, re-annotation sampling and agreement scoring | §3.1, Table 2 |
| `Re-Annotation Validation Sheet.xlsx` | The completed blind re-annotation of 200 records | §3.1, Table 2 |
| `Dataset_Change_Log.xlsx` | Record-level log of every change from v1.0 to v1.1 | §3.1 |
| `requirements.txt` | Python packages | — |
| `CITATION.cff` | Citation metadata | — |
| `LICENSE` / `LICENSE-CODE` | CC BY 4.0 (dataset) / MIT (code) | — |

## Reproducing the results

Install the packages: `pip install -r requirements.txt` (Python 3.10+; a GPU is recommended for the benchmark).
Run each script from the repository folder. Every script uses `GEOQUERYCA_DATASET_v1.1.csv` from the current folder, and the pipeline, coordinate and confidence-score scripts download it from this repository if it is not there. In Google Colab, paste a script into a cell (or use `%run <script>.py`); no Google Drive or manual upload is needed.

| Result | Command | Expected output |
|---|---|---|
| Benchmark and fine-tuning | `python geoqueryca_pipeline_github_release.py` | Tables 3–7 in `geoqueryca_outputs/`; split 1,408 / 590 |
| Field ablation | `python geoqueryca_field_ablation.py --csv GEOQUERYCA_DATASET_v1.1.csv` | baseline MRR 0.4399; `tableG3_field_ablation.tex` |
| Coordinate validation (≈2.5 h, needs internet) | `python coord_validation_github_release.py` | 1,450/1,904 matched; median 0.93 km; Fig. 2 |
| Confidence scores | `python confidence_score_github_release.py` | 2,001/2,001 identical |
| Re-annotation agreement | `python annotation_validation.py score --sheet "Re-Annotation Validation Sheet.xlsx"` | Cohen's κ = 0.81; 99.5% of labels confirmed; MAE 0.052 |

All benchmark settings (seed 42, B = 1,000 bootstrap resamples, batch size 32, maximum sequence length 256; fine-tuning: 3 epochs, batch size 16, 100 warm-up steps, 30% of landmarks held out) are defined in the `CONFIG` block at the top of `geoqueryca_pipeline_github_release.py`.
Coordinate lookups depend on the live Wikidata and GeoNames services, so a re-run may differ slightly from the published figures.

## License

- Dataset: Creative Commons Attribution 4.0 (CC BY 4.0) — see `LICENSE`
- Code: MIT — see `LICENSE-CODE`

## Citation

See `CITATION.cff`.
