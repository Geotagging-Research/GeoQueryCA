# GeoQueryCA

Multilingual, context-aware benchmark dataset for geographic disambiguation, geotagging and retrieval: **2,001 queries, 82 languages, 125 countries and territories, 1,904 unique landmarks**.

**Current version: v1.1** — the version used for every result in the paper. All changes from v1.0 are listed record by record in `dataset_construction/dataset_changelog.csv`.

## Dataset

File: `GEOQUERYCA_DATASET_v1.1.csv` (UTF-8 with BOM, comma-separated, 2,001 rows, 13 columns)
SHA-256: `0aa877a2ce18612b6faa147c0383d04106481c87f758bd11878eae10ee6e1ecc`

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
| `Confidence_Score` | float | Composite reliability score in [0, 1] (paper Eq. 1; see `dataset_construction/confidence_score.py`) | 0.92 |
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

```
GEOQUERYCA_DATASET_v1.1.csv            the dataset
src/
  geoqueryca_pipeline.py               profiling, zero-shot benchmark (6 models), bootstrap CIs,
                                       Holm-corrected significance tests, mBERT fine-tuning (paper §4, §6)
  geoqueryca_field_ablation.py         field ablation with mE5-large (paper §6.5, Table 8)
coordinate_validation/
  validate_coords3.py                  Wikidata (P625) / GeoNames lookup with ISO 3166 country check (paper §3.2)
  make_coord_fig.py                    Figure 2 from the validation output
  coord_validation_v3_colab.py         one-cell Google Colab version of the two scripts above
  coord_validation_v3full.csv          per-landmark results of the run reported in the paper
annotation_validation/
  annotation_validation.py             consistency checks, correction count, re-annotation sampling and scoring
  reannotation_sheet_completed.xlsx    the completed blind re-annotation (200 records)
  agreement_results.json               Cohen's κ, label confirmation, confidence MAE (paper Table 2)
  category_disagreements.csv           the category disagreements
dataset_construction/
  confidence_score.py                  recomputes Confidence_Score (paper Eq. 1); reproduces all 2,001 values
  fix_dataset.py                       applies every v1.0 → v1.1 correction, rule by rule
  dataset_changelog.csv                record-level log of all v1.0 → v1.1 changes
requirements.txt
```

## Reproducing the results

Install: `pip install -r requirements.txt` (Python 3.10+; a GPU is recommended for the benchmark).
Run every command from the repository root, next to `GEOQUERYCA_DATASET_v1.1.csv`.

| Paper result | Command |
|---|---|
| Tables 3–7, Figs. 3–5, Geo-wMRR, significance tests, fine-tuning (1,408 train / 590 test) | `python src/geoqueryca_pipeline.py` |
| Table 8 (field ablation) | `python src/geoqueryca_field_ablation.py --csv GEOQUERYCA_DATASET_v1.1.csv` |
| §3.2, Table 2 (coordinates), Fig. 2 | `python coordinate_validation/validate_coords3.py` then `python coordinate_validation/make_coord_fig.py coord_validation_v3full.csv` (about 2.5 h; needs internet) |
| Confidence scores (Eq. 1) | `python dataset_construction/confidence_score.py` |
| Re-annotation agreement (Table 2) | `python annotation_validation/annotation_validation.py score --sheet annotation_validation/reannotation_sheet_completed.xlsx` |
| Correction count (Table 2) | `python annotation_validation/annotation_validation.py diff --old <pre-validation file>` |
| v1.0 → v1.1 corrections | `python dataset_construction/fix_dataset.py --data GEOQUERYCA_DATASET_V1.0.csv` (v1.0 file from the V1.0 release) |

All benchmark settings (seed 42, B = 1,000, batch size 32, max. sequence length 256, fine-tuning: 3 epochs, batch size 16, 100 warm-up steps, 30% of landmarks held out) are defined in the `CONFIG` block at the top of `src/geoqueryca_pipeline.py`.
Coordinate lookups depend on the live Wikidata and GeoNames services, so a re-run may differ slightly from the archived `coord_validation_v3full.csv`.

## License

- Dataset: Creative Commons Attribution 4.0 (CC BY 4.0) — see `LICENSE`
- Code: MIT — see `src/LICENSE-CODE`

## Citation

See `CITATION.cff`.
