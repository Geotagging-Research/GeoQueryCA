"""
GeoQueryCA — Confidence score  (paper §4.3, Eq. 1)
=================================================
Recomputes the Confidence_Score column of the released dataset:

    CS_i = 0.30 s_L + 0.25 s_C + 0.25 s_D + 0.10 s_V + 0.10 s_A

  s_L  language resource level   1.00 high · 0.67 medium · 0.33 low
                                 (tiers after NLLB Team et al., 2022; see LANG_TIER)
  s_C  coordinate validity       1.00 both valid · 0.60 one · 0.20 neither
                                 (valid = parses to a number in range; DMS or decimal)
  s_D  cue richness              1.00 if ≥200 chars and ≥6 comma-separated terms
                                 0.80 if ≥120 and ≥4 · 0.60 if ≥60 and ≥2 · 0.20 otherwise
  s_V  category verifiability    1.00 physically bounded · 0.70 intermediate · 0.40 abstract
                                 (see CATEGORY_TIER)
  s_A  name uniqueness           1.00 unique · 0.70 ×2 · 0.50 ×3 · 0.30 ×4+
                                 (occurrences of the Monument name in the dataset)

Usage
-----
  python confidence_score.py                                   # verify the released scores
                                                               # (downloads the dataset if missing)
  python confidence_score.py --data in.csv --write out.csv                     # (re)score

Verification compares the recomputed score (rounded to 2 dp) with the
Confidence_Score column and reports any mismatch.
"""
import argparse, re, sys
import numpy as np
import pandas as pd

W = {"L": 0.30, "C": 0.25, "D": 0.25, "V": 0.10, "A": 0.10}

# ── s_L: language resource tiers (high = 1.00, medium = 0.67, low = 0.33) ──
HIGH = ["Arabic", "Bengali", "Chinese", "Dutch", "English", "French", "German",
        "Hindi", "Indonesian", "Italian", "Japanese", "Korean", "Mandarin",
        "Polish", "Portuguese", "Russian", "Spanish", "Thai", "Turkish",
        "Vietnamese"]
MEDIUM = ["Afrikaans", "Albanian", "Amharic", "Assamese", "Azerbaijani",
          "Bosnian", "Bulgarian", "Burmese", "Croatian", "Czech", "Danish",
          "Estonian", "Filipino", "Finnish", "Georgian", "Greek", "Gujarati",
          "Hebrew", "Hungarian", "Icelandic", "Kannada", "Khmer", "Kyrgyz",
          "Lao", "Latvian", "Lithuanian", "Macedonian", "Malayalam", "Marathi",
          "Meitei", "Nepali", "Norwegian", "Odia", "Persian", "Punjabi",
          "Romanian", "Serbian", "Sinhala", "Slovenian", "Swahili", "Swedish",
          "Tajik", "Tamil", "Telugu", "Turkmen", "Urdu", "Uzbek"]
LOW = ["Angami", "Apatani", "Dogri", "Garo", "Hawaiian", "Khasi", "Konkani",
       "Ladakhi", "Maori", "Mizo", "Monpa", "Montenegrin", "Nagamese",
       "Rajasthani", "Tibetan"]
LANG_TIER = {**{l: 1.00 for l in HIGH}, **{l: 0.67 for l in MEDIUM},
             **{l: 0.33 for l in LOW}}

# ── s_V: category verifiability ──────────────────────────────────────────────
CATEGORY_TIER = {
    # physically bounded, single identifiable site → 1.00
    "Historical Site": 1.00, "Historical": 1.00, "Monument": 1.00,
    "Religious Site": 1.00, "Palace": 1.00, "Castle": 1.00, "Museum": 1.00,
    "Landmark": 1.00, "National Park": 1.00, "Educational Institution": 1.00,
    "University": 1.00, "Research Institution": 1.00,
    # intermediate: extended area or event → 0.70
    "Natural Attraction": 0.70, "Natural Phenomenon": 0.70,
    "Tourist Attraction": 0.70, "Attraction": 0.70, "Festival": 0.70,
    # abstract / experiential → 0.40
    "Tourist Experience": 0.40,
}


def s_coord(v, lo, hi):
    """1 if the value parses to a number in [lo, hi] (decimal or DMS)."""
    if pd.isna(v):
        return 0
    try:
        x = float(v)
    except (TypeError, ValueError):
        m = re.match(r"\s*(\d+)\D+(\d+)\D+(\d+(?:\.\d+)?)\D*([NSEW])", str(v))
        if not m:
            return 0
        x = int(m[1]) + int(m[2]) / 60 + float(m[3]) / 3600
        x = -x if m[4] in "SW" else x
    return int(lo <= x <= hi)


def s_cues(c):
    c = "" if pd.isna(c) else str(c)
    n_chars = len(c)
    n_terms = len([t for t in c.split(",") if t.strip()])
    if n_chars >= 200 and n_terms >= 6: return 1.00
    if n_chars >= 120 and n_terms >= 4: return 0.80
    if n_chars >= 60 and n_terms >= 2:  return 0.60
    return 0.20


def score(df):
    out = pd.DataFrame(index=df.index)
    lang = df["Query_Language"].astype(str).str.strip()
    cat = df["Category"].astype(str).str.strip()
    unknown_l = sorted(set(lang) - set(LANG_TIER))
    unknown_c = sorted(set(cat) - set(CATEGORY_TIER) - {"Historical"})
    if unknown_l or unknown_c:
        sys.exit(f"Add a tier for: languages {unknown_l}  categories {unknown_c}")
    out["s_L"] = lang.map(LANG_TIER)
    nvalid = (df["Latitude"].map(lambda v: s_coord(v, -90, 90)) +
              df["Longitude"].map(lambda v: s_coord(v, -180, 180)))
    out["s_C"] = nvalid.map({2: 1.00, 1: 0.60, 0: 0.20})
    out["s_D"] = df["Disambiguation_Cues"].map(s_cues)
    cat_key = cat.where(cat != "Historical", "Historical")
    out["s_V"] = cat_key.map(CATEGORY_TIER)
    n = df["Monument"].map(df["Monument"].value_counts())
    out["s_A"] = n.map(lambda k: 1.00 if k == 1 else 0.70 if k == 2
                       else 0.50 if k == 3 else 0.30)
    out["CS"] = sum(W[k] * out[f"s_{k}"] for k in W)
    out["CS_2dp"] = (out["CS"] + 1e-9).round(2)
    return out


DATA_URL = ("https://raw.githubusercontent.com/Geotagging-Research/GeoQueryCA/"
            "main/GEOQUERYCA_DATASET_v1.1.csv")


def _find_or_download(path):
    """Use the file if present (also checks data/ and /content); if the
    released dataset is missing, download it from the GitHub repository."""
    import os, urllib.request
    for p in [path, os.path.join("data", path), os.path.join("/content", path)]:
        if os.path.exists(p):
            return p
    if os.path.basename(path) == "GEOQUERYCA_DATASET_v1.1.csv":
        print(f"'{path}' not found — downloading it from GitHub …")
        urllib.request.urlretrieve(DATA_URL, "GEOQUERYCA_DATASET_v1.1.csv")
        return "GEOQUERYCA_DATASET_v1.1.csv"
    sys.exit(f"'{path}' not found. Put it next to this script or pass --data <path>.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="GEOQUERYCA_DATASET_v1.1.csv")
    p.add_argument("--write", help="write a copy with Confidence_Score recomputed")
    a, _ = p.parse_known_args()
    a.data = _find_or_download(a.data)
    df = pd.read_csv(a.data, encoding="utf-8-sig").rename(columns={"Landmark": "Monument"})
    sc = score(df)

    print(f"\nConfidence score — {len(df):,} records")
    for k in W:
        vc = sc[f"s_{k}"].value_counts().sort_index(ascending=False)
        print(f"  s_{k} (w={W[k]:.2f}): " +
              ", ".join(f"{v:.2f}×{n}" for v, n in vc.items()))
    print(f"  CS: min {sc.CS_2dp.min():.2f}  max {sc.CS_2dp.max():.2f}  "
          f"mean {sc.CS_2dp.mean():.3f}  sd {sc.CS_2dp.std():.3f}")

    if "Confidence_Score" in df:
        diff = (sc["CS_2dp"] - df["Confidence_Score"].astype(float)).abs() > 0.005
        print(f"\n  Verification vs released Confidence_Score: "
              f"{(~diff).sum():,}/{len(df):,} identical")
        if diff.any():
            show = df.loc[diff, ["Monument", "Query_Language", "Category",
                                 "Confidence_Score"]].assign(recomputed=sc.loc[diff, "CS_2dp"])
            print(show.head(20).to_string())

    if a.write:
        df = df.copy()
        df["Confidence_Score"] = sc["CS_2dp"]
        df.to_csv(a.write, index=False, encoding="utf-8-sig")
        print(f"\n  → wrote {a.write}")


if __name__ == "__main__":
    main()
