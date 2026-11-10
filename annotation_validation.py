"""
GeoQueryCA — Annotation validation  (Table 2, "Three-phase validation" row)
==========================================================================
Reproduces the annotation-validation numbers reported in the paper.

  check   Phase 1 — automated consistency checks on the released dataset
  diff    Correction count: pre-validation version vs released version
  sample  Phase 2 — draw a stratified re-annotation sample and export a
          blind annotation sheet (gold labels hidden) for a second annotator
  score   Phase 2 — Cohen's kappa (Category), confidence-score MAE and
          label-validity agreement from the returned annotation sheet

(Phase 3, coordinate cross-referencing, is validate_coords2.py + Figure 2.)

Usage
-----
  python annotation_validation.py check  --data GEOQUERYCA_DATASET_v1.1.csv
  python annotation_validation.py diff   --data GEOQUERYCA_DATASET_v1.1.csv \
                                         --old  GEOQUERYCA_minor_issues_flagged.xlsx
  python annotation_validation.py sample --data GEOQUERYCA_DATASET_v1.1.csv --n 200
  python annotation_validation.py score  --data GEOQUERYCA_DATASET_v1.1.csv \
                                         --sheet reannotation_sheet_filled.xlsx

Outputs go to ./annotation_validation_outputs/.  Seed fixed (42).
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

SEED = 42
OUT = "annotation_validation_outputs"

ESSENTIAL = ["Monument", "City", "Latitude", "Longitude", "Country", "Category",
             "Contextual_Query", "Query_Language", "Disambiguation_Cues",
             "Confidence_Score", "Region", "Nearby_Attractions",
             "Cultural_Significance"]
TEXT_FIELDS = ["Monument", "City", "Country", "Category", "Query_Language",
               "Region", "Disambiguation_Cues", "Nearby_Attractions",
               "Cultural_Significance"]
REGIONS = {"Asia", "Europe", "Americas", "Africa", "Oceania"}
# UTF-8 text decoded as Latin-1 / cp1252 ("Ã©" for "é", "â€™" for "’")
MOJIBAKE = re.compile(r"[ÃÂ][\x80-\xBF -ÿ]|â€[™œsd\x9d]")

INDIAN = {"Hindi", "Marathi", "Bengali", "Tamil", "Telugu", "Kannada",
          "Malayalam", "Odia", "Oriya", "Punjabi", "Gujarati", "Assamese",
          "Urdu", "Sanskrit", "Sindhi", "Konkani", "Maithili", "Nepali",
          "Sinhala"}
EUROPEAN = {"Spanish", "French", "German", "Italian", "Portuguese", "Dutch",
            "Polish", "Romanian", "Czech", "Greek", "Swedish", "Russian",
            "Finnish", "Hungarian", "Danish", "Norwegian", "Bulgarian",
            "Croatian", "Slovak", "Catalan", "Ukrainian"}


def lang_group(l):
    if l == "English":  return "English"
    if l in INDIAN:     return "Indian vernaculars"
    if l in EUROPEAN:   return "European"
    return "Others"


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns={"Landmark": "Monument"})
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string")
    return df


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def _save_json(obj, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — automated consistency checks
# ════════════════════════════════════════════════════════════════════════════
def cmd_check(a):
    df = load(a.data)
    N = len(df)
    issues = []          # (row_index, check, detail)

    def flag(mask, check, detail_fn):
        for i in np.where(np.asarray(mask, dtype=bool))[0]:
            issues.append((int(i), check, detail_fn(df.iloc[i])))

    # 1. schema
    missing_cols = [c for c in ESSENTIAL if c not in df.columns]

    # 2. missing / blank values
    for c in [c for c in ESSENTIAL if c in df.columns]:
        blank = df[c].isna() | (df[c].astype("string").str.strip() == "")
        flag(blank, f"missing_{c}", lambda r, c=c: f"{c} is empty")

    # 3. leading/trailing whitespace
    for c in TEXT_FIELDS + ["Contextual_Query"]:
        if c in df:
            s = df[c].astype("string")
            flag(s.notna() & (s != s.str.strip()), "whitespace",
                 lambda r, c=c: f"{c} has leading/trailing spaces")

    # 4. encoding corruption (mojibake)
    for c in TEXT_FIELDS + ["Contextual_Query"]:
        if c in df:
            flag(df[c].astype(str).str.contains(MOJIBAKE), "mojibake",
                 lambda r, c=c: f"{c}: {str(r[c])[:60]}")

    # 5. coordinates
    lat = pd.to_numeric(df["Latitude"], errors="coerce").astype(float)
    lon = pd.to_numeric(df["Longitude"], errors="coerce").astype(float)
    flag(lat.isna() | lon.isna(), "coord_not_a_number",
         lambda r: f"({r['Latitude']!r}, {r['Longitude']!r})")
    flag(~lat.between(-90, 90) | ~lon.between(-180, 180), "coord_out_of_range",
         lambda r: f"({r['Latitude']}, {r['Longitude']})")
    flag((lat == 0) & (lon == 0), "coord_null_island", lambda r: "(0, 0)")

    # 6. confidence score range
    cs = pd.to_numeric(df["Confidence_Score"], errors="coerce")
    flag(~cs.between(0, 1), "confidence_out_of_range",
         lambda r: str(r["Confidence_Score"]))

    # 7. region vocabulary
    flag(~df["Region"].isin(REGIONS), "region_unknown", lambda r: str(r["Region"]))

    # 8. exact duplicate records
    flag(df.duplicated(keep="first"), "duplicate_row", lambda r: r["Monument"])

    # 9. duplicate queries — same gold (redundant) vs different gold (conflict)
    q = df["Contextual_Query"].map(norm)
    gold = df["Monument"].map(norm) + " | " + df["City"].map(norm)
    dup_groups = {"consistent": 0, "conflict": 0}
    for _, idx in q.groupby(q).groups.items():
        if len(idx) < 2:
            continue
        idx = list(idx)
        if gold.loc[idx].nunique() == 1:
            dup_groups["consistent"] += 1
            flag(df.index.isin(idx[1:]), "duplicate_query_same_gold",
                 lambda r: r["Monument"])
        else:
            dup_groups["conflict"] += 1
            flag(df.index.isin(idx), "duplicate_query_conflicting_gold",
                 lambda r: f"{r['Monument']} ({r['City']})")

    # 10. design criterion: query must not name the landmark
    def names_landmark(r):
        m = norm(r["Monument"])
        return len(m) >= 4 and m in norm(r["Contextual_Query"])
    flag(df.apply(names_landmark, axis=1), "query_names_landmark",
         lambda r: r["Monument"])

    # 11. same landmark → inconsistent country / region / coordinates
    key = df["Monument"].map(norm) + " | " + df["City"].map(norm)
    for col in ["Country", "Region"]:
        n_u = df.groupby(key)[col].transform("nunique")
        flag(n_u > 1, f"inconsistent_{col.lower()}_same_landmark",
             lambda r, col=col: f"{r['Monument']}: {r[col]}")
    spread = (df.assign(_lat=lat, _lon=lon).groupby(key)[["_lat", "_lon"]]
                .transform(lambda s: s.max() - s.min()).max(axis=1))
    flag(spread > 0.05, "inconsistent_coords_same_landmark",   # > ~5 km
         lambda r: r["Monument"])

    # 12. vocabulary hygiene (reported, not counted as errors)
    cat_counts = df["Category"].value_counts()
    rare_categories = cat_counts[cat_counts < 5].to_dict()
    lang_aliases = [p for p in [("Chinese", "Mandarin"), ("Odia", "Oriya")]
                    if set(p) <= set(df["Query_Language"].unique())]

    # ── summary ────────────────────────────────────────────────────────────
    iss = pd.DataFrame(issues, columns=["row_index", "check", "detail"])
    iss["csv_line"] = iss["row_index"] + 2            # header = line 1
    per_check = iss.groupby("check")["row_index"].nunique().sort_values(ascending=False)
    rows_flagged = int(iss["row_index"].nunique()) if len(iss) else 0

    summary = {
        "dataset": os.path.basename(a.data), "n_records": N,
        "missing_columns": missing_cols,
        "records_flagged": rows_flagged,
        "records_flagged_pct": round(100 * rows_flagged / N, 2),
        "records_per_check": per_check.to_dict(),
        "duplicate_query_groups": dup_groups,
        "rare_categories_lt5": rare_categories,
        "language_label_aliases_present": lang_aliases,
    }
    os.makedirs(OUT, exist_ok=True)
    iss.sort_values(["check", "row_index"]).to_csv(
        os.path.join(OUT, "consistency_issues.csv"), index=False, encoding="utf-8-sig")
    _save_json(summary, "consistency_summary.json")

    print(f"\nConsistency checks — {N:,} records")
    if missing_cols:
        print(f"  MISSING COLUMNS: {missing_cols}")
    print(f"  Records with ≥1 issue: {rows_flagged} ({summary['records_flagged_pct']}%)")
    for k, v in per_check.items():
        print(f"    {k:<40} {v:>5}")
    if not len(per_check):
        print("    (no issues)")
    print(f"  Duplicate-query groups: {dup_groups}")
    print(f"  Rare categories (<5 records): {rare_categories}")
    if lang_aliases:
        print(f"  Language labels that may be aliases: {lang_aliases}")
    print(f"\n  → {OUT}/consistency_issues.csv, consistency_summary.json")


# ════════════════════════════════════════════════════════════════════════════
# Correction count — earlier (pre-validation) version vs released version
# ════════════════════════════════════════════════════════════════════════════
def _load_old(path):
    if path.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None)
        name = next((s for s in sheets if "Full_Dataset" in s), None)
        if name is None:
            sys.exit(f"No 'Full_Dataset…' sheet in {path}")
        old = sheets[name]
    else:
        old = pd.read_csv(path, encoding="utf-8-sig")
    old = old.loc[:, ~old.columns.astype(str).str.match(r"^Unnamed")]
    return old


def cmd_diff(a):
    new = load(a.data)
    old = _load_old(a.old)
    fields = [c for c in TEXT_FIELDS if c in old.columns and c in new.columns]

    # Match records on the query text (the record's identity in GeoQueryCA).
    # Queries occurring more than once in either version are matched by order.
    def keyed(d):
        d = d.copy()
        d["_q"] = d["Contextual_Query"].map(norm)
        d["_k"] = d["_q"] + "#" + d.groupby("_q").cumcount().astype(str)
        return d.set_index("_k")
    o, n = keyed(old), keyed(new)
    common = o.index.intersection(n.index)

    # Each change is classified, because not every edit is an error fix:
    #   correction     — a wrong/missing value fixed (labels, metadata, text)
    #   normalisation  — Region vocabulary mapped to the 5 macro-regions
    #   cue_deleaking  — Disambiguation_Cues edited ONLY by deleting words
    #                    (systematic removal of place names that leaked the answer)
    tok = lambda s: set(re.findall(r"\w+", s))
    changed_by_field = {}
    rows_by_class = {"correction": set(), "normalisation": set(), "cue_deleaking": set()}
    changes = []
    for c in fields:
        a_ = o.loc[common, c].map(lambda v: "" if pd.isna(v) else norm(v))
        b_ = n.loc[common, c].map(lambda v: "" if pd.isna(v) else norm(v))
        diff = a_ != b_
        changed_by_field[c] = int(diff.sum())
        for k in common[diff.values]:
            if c == "Region":
                cls = "normalisation"
            elif c == "Disambiguation_Cues" and a_[k] and tok(b_[k]) <= tok(a_[k]):
                cls = "cue_deleaking"
            else:
                cls = "correction"
            rows_by_class[cls].add(k)
            changes.append({"field": c, "class": cls,
                            "old": str(o.at[k, c]), "new": str(n.at[k, c]),
                            "query": str(n.at[k, "Contextual_Query"])[:120]})
    changed_rows = set().union(*rows_by_class.values())
    corrected = rows_by_class["correction"]
    ch = pd.DataFrame(changes)
    examples = ch.groupby(["field", "class"]).head(2)[["field", "class", "old", "new"]] \
                 .map(lambda v: str(v)[:80]).to_dict("records") if len(ch) else []

    conf_changed = None
    if "Confidence_Score" in old and "Confidence_Score" in new:
        ca = pd.to_numeric(o.loc[common, "Confidence_Score"], errors="coerce")
        cb = pd.to_numeric(n.loc[common, "Confidence_Score"], errors="coerce")
        conf_changed = int((ca.round(4) != cb.round(4)).sum())

    res = {
        "old_version": os.path.basename(a.old), "n_old": len(old),
        "new_version": os.path.basename(a.data), "n_new": len(new),
        "matched_records": int(len(common)),
        "only_in_old_removed": int(len(o.index.difference(n.index))),
        "only_in_new_added": int(len(n.index.difference(o.index))),
        "matched_records_edited_any": len(changed_rows),
        "records_by_class": {k: len(v) for k, v in rows_by_class.items()},
        "records_corrected": len(corrected),
        "records_corrected_pct_of_matched": round(100 * len(corrected) / max(len(common), 1), 2),
        "records_corrected_pct_of_released": round(100 * len(corrected) / len(new), 2),
        "corrections_by_field": (ch[ch["class"] == "correction"]["field"].value_counts().to_dict()
                                 if len(ch) else {}),
        "changes_by_field_all": changed_by_field,
        "confidence_scores_changed (formula re-scoring, not a correction)": conf_changed,
        "examples": examples,
    }
    _save_json(res, "correction_diff.json")
    os.makedirs(OUT, exist_ok=True)
    ch.to_csv(os.path.join(OUT, "all_changes.csv"), index=False, encoding="utf-8-sig")

    print(f"\nCorrection diff — {res['n_old']:,} (old) vs {res['n_new']:,} (released)")
    print(f"  Matched on query text: {res['matched_records']:,}   "
          f"removed: {res['only_in_old_removed']}   added: {res['only_in_new_added']}")
    print(f"  Records edited (any change): {len(changed_rows)}")
    for k, v in res["records_by_class"].items():
        print(f"    {k:<16} {v:>5} records")
    print(f"  → RECORDS CORRECTED: {len(corrected)} "
          f"({res['records_corrected_pct_of_matched']}% of matched, "
          f"{res['records_corrected_pct_of_released']}% of released)")
    for k, v in res["corrections_by_field"].items():
        print(f"      {k:<24} {v:>5}")
    if conf_changed is not None:
        print(f"  Confidence scores re-computed (not counted): {conf_changed}")
    print(f"\n  → {OUT}/correction_diff.json, all_changes.csv")


# ════════════════════════════════════════════════════════════════════════════
# Phase 2a — stratified blind re-annotation sample
# ════════════════════════════════════════════════════════════════════════════
def cmd_sample(a):
    df = load(a.data).reset_index().rename(columns={"index": "row_index"})
    df["group"] = df["Query_Language"].map(lang_group)
    strata = df["group"] + " | " + df["Region"]
    frac = a.n / len(df)
    # proportional allocation, at least 1 per stratum, then trim/top-up to n
    alloc = (strata.value_counts() * frac).round().clip(lower=1).astype(int)
    rng = np.random.default_rng(SEED)
    picks = []
    for s, k in alloc.items():
        idx = df.index[strata == s].to_numpy()
        picks.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    picks = list(dict.fromkeys(picks))
    if len(picks) > a.n:
        picks = list(rng.choice(picks, size=a.n, replace=False))
    elif len(picks) < a.n:
        rest = np.setdiff1d(df.index.to_numpy(), picks)
        picks += list(rng.choice(rest, size=a.n - len(picks), replace=False))
    s = df.loc[sorted(picks)].sample(frac=1, random_state=SEED)   # shuffle order

    categories = sorted({str(c).strip() for c in df["Category"].dropna()})
    sheet = pd.DataFrame({
        "Item_ID": range(1, len(s) + 1),
        # Excel row number of the record in the dataset CSV (header = row 1)
        "Dataset_Row (Excel)": s["row_index"].values + 2,
        "Contextual_Query": s["Contextual_Query"].values,
        "Query_Language": s["Query_Language"].values,
        "Proposed_Landmark": s["Monument"].values,
        "City": s["City"].values,
        "Country": s["Country"].values,
        # ── to be filled by the second annotator ──
        "A2_Query_Matches_Landmark (Yes/No)": "",
        "A2_Category": "",
        "A2_Confidence (0-1)": "",
        "A2_Comment": "",
    })
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "reannotation_sheet.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sheet.to_excel(xw, sheet_name="Annotate", index=False)
        pd.DataFrame({"Allowed categories": categories}).to_excel(
            xw, sheet_name="Categories", index=False)
        pd.DataFrame({"Instructions": [
            "Work alone; do not look at the released dataset or the first annotator's labels.",
            "A2_Query_Matches_Landmark: Yes if the query unambiguously describes the Proposed_Landmark, else No.",
            "A2_Category: choose exactly one value from the 'Categories' sheet.",
            "A2_Confidence: how certain is it that the query identifies this landmark? "
            "0 = not at all, 1 = certain. Use two decimals (e.g. 0.85), following the rubric in the paper (§3).",
            "Leave a short comment for any No or uncertain case.",
        ]}).to_excel(xw, sheet_name="Instructions", index=False)
        ws = xw.sheets["Annotate"]
        from openpyxl.worksheet.datavalidation import DataValidation
        n = len(sheet) + 1
        dv_yn = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
        # Inline list (≤255 chars) works in Excel, LibreOffice and Google
        # Sheets; fall back to a sheet range (no leading "=") if too long.
        _inline = '"' + ",".join(categories) + '"'
        dv_cat = DataValidation(type="list", allow_blank=True,
                                formula1=_inline if len(_inline) <= 255 else
                                f"Categories!$A$2:$A${len(categories)+1}")
        dv_cf = DataValidation(type="decimal", operator="between",
                               formula1="0", formula2="1", allow_blank=True)
        for dv, col in [(dv_yn, "H"), (dv_cat, "I"), (dv_cf, "J")]:
            ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n}")
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        # Thin borders on every cell: a solid fill hides Excel's gridlines,
        # so without them the rows in the shaded columns are not visible.
        _s = Side(style="thin", color="BFBFBF")
        for row in ws.iter_rows(min_row=1, max_row=n, min_col=1, max_col=11):
            for cell in row:
                cell.border = Border(left=_s, right=_s, top=_s, bottom=_s)
        for col, w in zip("ABCDEFGHIJK", [8, 9, 70, 14, 30, 18, 14, 22, 26, 18, 30]):
            ws.column_dimensions[col].width = w
        ws.row_dimensions[1].height = 36
        fill = PatternFill("solid", fgColor="FFF2CC")       # columns to fill in
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            if cell.column_letter in "HIJK":
                cell.fill = fill
        for r in range(2, n + 1):
            ws[f"C{r}"].alignment = Alignment(wrap_text=True, vertical="top")
            for col in "HIJK":
                ws[f"{col}{r}"].fill = fill
        ws.freeze_panes = "C2"

    comp = (s.assign(group=s["Query_Language"].map(lang_group))
              .groupby(["group", "Region"]).size().rename("n").reset_index())
    comp.to_csv(os.path.join(OUT, "sample_composition.csv"), index=False)
    print(f"\nRe-annotation sample: {len(s)} records (seed {SEED}), "
          f"{strata.nunique()} language-group × region strata")
    print(comp.to_string(index=False))
    print(f"\n  → {path}  (send to the second annotator)")


# ════════════════════════════════════════════════════════════════════════════
# Phase 2b — agreement statistics from the returned sheet
# ════════════════════════════════════════════════════════════════════════════
def cohen_kappa(y1, y2):
    y1, y2 = np.asarray(y1), np.asarray(y2)
    labels = np.union1d(y1, y2)
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels),) * 2)
    for p, q in zip(y1, y2):
        m[idx[p], idx[q]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0, float(po)


def cmd_score(a):
    df = load(a.data)
    sh = pd.read_excel(a.sheet, sheet_name="Annotate")
    # Accept both sheet versions: "Dataset_Row (Excel)" (header = row 1)
    # or the older 0-based "row_index".
    if any("dataset_row" in str(c).lower() for c in sh.columns):
        _rc = next(c for c in sh.columns if "dataset_row" in str(c).lower())
        sh["row_index"] = sh[_rc].astype(int) - 2
    gold = df.loc[sh["row_index"].astype(int)].reset_index(drop=True)
    sh = sh.reset_index(drop=True)
    # Find the annotator columns by keyword, so renamed headers
    # (e.g. "Annotator_Category" instead of "A2_Category") still work.
    def _col(*keys):
        for c in sh.columns:
            if all(k.lower() in str(c).lower() for k in keys):
                return c
        sys.exit(f"Column containing {keys} not found in {a.sheet}")
    sh = sh.rename(columns={_col("matches"): "A2_Query_Matches_Landmark (Y/N)",
                            _col("category"): "A2_Category",
                            _col("confidence"): "A2_Confidence (0-1)"})
    res = {"n_items": len(sh)}

    # Category — Cohen's kappa
    c2 = sh["A2_Category"].astype("string").str.strip()
    ok = c2.notna() & (c2 != "")
    k, po = cohen_kappa(gold.loc[ok, "Category"].map(norm), c2[ok].map(norm))
    res["category"] = {"n": int(ok.sum()), "cohen_kappa": round(k, 4),
                       "observed_agreement": round(po, 4)}

    # Confidence — MAE vs released score
    c = pd.to_numeric(sh["A2_Confidence (0-1)"], errors="coerce")
    okc = c.notna()
    err = (c[okc] - gold.loc[okc, "Confidence_Score"].astype(float)).abs()
    rng = np.random.default_rng(SEED)
    boot = [err.sample(frac=1, replace=True, random_state=int(s)).mean()
            for s in rng.integers(0, 2**31, 1000)]
    # Spearman rank correlation: does the formula score track the human
    # judgement of identifiability? (The formula measures entry reliability
    # from dataset-intrinsic fields, so rank agreement is the fairer test.)
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(c[okc], gold.loc[okc, "Confidence_Score"].astype(float))
    res["confidence"] = {"n": int(okc.sum()), "MAE": round(float(err.mean()), 4),
                         "MAE_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                      round(float(np.percentile(boot, 97.5)), 4)],
                         "spearman_rho": round(float(rho), 4),
                         "spearman_p": float(p_rho),
                         "mean_annotator": round(float(c[okc].mean()), 4),
                         "mean_released": round(float(gold.loc[okc, "Confidence_Score"].mean()), 4)}

    # Category confusion pairs and per-row disagreement list (for review)
    g_cat = gold.loc[ok, "Category"].astype(str).str.strip()
    a_cat = c2[ok].astype(str)
    mism = g_cat.map(norm) != a_cat.map(norm)
    res["category"]["n_disagree"] = int(mism.sum())
    res["category"]["confusions"] = (
        (g_cat[mism] + " → " + a_cat[mism]).value_counts().to_dict())
    dis = pd.DataFrame({
        "Item_ID": sh.loc[ok, sh.columns[0]][mism].values,
        "Dataset_Row (Excel)": (sh.loc[ok, "row_index"][mism].astype(int) + 2).values,
        "Proposed_Landmark": gold.loc[ok, "Monument"][mism].values,
        "Released_Category": g_cat[mism].values,
        "Annotator_Category": a_cat[mism].values,
    })
    os.makedirs(OUT, exist_ok=True)
    dis.to_csv(os.path.join(OUT, "category_disagreements.csv"),
               index=False, encoding="utf-8-sig")

    # Label validity — does the query identify the gold landmark?
    yn = (sh["A2_Query_Matches_Landmark (Y/N)"].astype("string").str.strip().str.upper()
            .replace({"YES": "Y", "NO": "N"}))          # accepts Yes/No or Y/N
    oky = yn.isin(["Y", "N"])
    res["label_validity"] = {"n": int(oky.sum()),
                             "pct_confirmed": round(100 * float((yn[oky] == "Y").mean()), 2),
                             "rows_rejected_excel_row": (sh.loc[oky & (yn == "N"), "row_index"].astype(int) + 2).tolist()}

    _save_json(res, "agreement_results.json")
    print(f"\nRe-annotation agreement ({res['n_items']} items)")
    print(f"  Category  Cohen's κ = {res['category']['cohen_kappa']:.2f}  "
          f"(observed agreement {100*res['category']['observed_agreement']:.1f}%, "
          f"n={res['category']['n']})")
    print(f"  Confidence MAE = {res['confidence']['MAE']:.3f}  "
          f"95% CI {res['confidence']['MAE_ci95']}  (n={res['confidence']['n']})")
    print(f"  Confidence Spearman ρ = {res['confidence']['spearman_rho']:.3f}  "
          f"(p = {res['confidence']['spearman_p']:.3g}); means: annotator "
          f"{res['confidence']['mean_annotator']:.3f} vs released "
          f"{res['confidence']['mean_released']:.3f}")
    print(f"  Category disagreements: {res['category']['n_disagree']} "
          f"→ {OUT}/category_disagreements.csv")
    for pair, n in list(res["category"]["confusions"].items())[:10]:
        print(f"      {pair:<50} {n:>3}")
    print(f"  Gold label confirmed: {res['label_validity']['pct_confirmed']:.1f}% "
          f"(n={res['label_validity']['n']})")
    print(f"\n  → {OUT}/agreement_results.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ["check", "diff", "sample", "score"]:
        s = sp.add_parser(name)
        s.add_argument("--data", default="GEOQUERYCA_DATASET_v1.1.csv")
        if name == "diff":
            s.add_argument("--old", required=True,
                           help="pre-validation version (CSV, or the flagged .xlsx)")
        if name == "sample":
            s.add_argument("--n", type=int, default=200)
        if name == "score":
            s.add_argument("--sheet", required=True)
    a, _ = p.parse_known_args()      # parse_known_args: also safe inside Colab
    {"check": cmd_check, "diff": cmd_diff,
     "sample": cmd_sample, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
