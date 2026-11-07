"""
geoqueryca_field_ablation.py
============================
Field-ablation study for the GeoQueryCA paper (new Section 6.7).

Measures the retrieval contribution of the dataset's annotation fields by
re-running the strongest zero-shot model (mE5-large) under six input
configurations:

  baseline     Q = Contextual_Query                P = Cultural_Significance
  +cues        Q = Query + Disambiguation_Cues     P = Cultural_Significance
  cues-only    Q = Disambiguation_Cues             P = Cultural_Significance
  +nearby      Q = Contextual_Query                P = CS + Nearby_Attractions
  nearby-only  Q = Contextual_Query                P = Nearby_Attractions
  full         Q = Query + Cues                    P = CS + Nearby_Attractions

Each ablation is compared against the baseline with a two-sided paired
bootstrap on per-query reciprocal rank (B=1,000, H0-centred), Holm-corrected
across the 5 comparisons. Metrics reported overall and for the Indian
vernacular subgroup.

Run in Colab (GPU runtime) in a cell:

    !pip -q install sentence-transformers
    %run geoqueryca_field_ablation.py --csv /content/GeoQueryCA.csv

Outputs (working directory):
    ablation_results.json
    tableG3_field_ablation.tex
    ablation_numbers.md
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 42
BOOT_B = 1000
MODEL_NAME = "intfloat/multilingual-e5-large"
PREFIX_Q, PREFIX_P = "query: ", "passage: "
BATCH = 64

# ── column aliases (mirrors geoqueryca_pipeline_v3) ─────────────────────────
COL_ALIASES = {
    "Lat": "Latitude", "lat": "Latitude", "Lon": "Longitude",
    "Long": "Longitude", "lon": "Longitude", "long": "Longitude",
    "Cultural_Sig": "Cultural_Significance",
    "Query": "Contextual_Query", "Language": "Query_Language",
    "Point_of_Interest": "Monument", "point_of_interest": "Monument",
    "POI": "Monument",
}

INDIAN_VERNACULARS = {"Hindi", "Marathi", "Bengali", "Tamil", "Telugu",
                      "Kannada", "Malayalam", "Odia", "Oriya", "Punjabi",
                      "Gujarati", "Assamese", "Urdu", "Sanskrit", "Sindhi",
                      "Konkani", "Maithili", "Nepali", "Sinhala"}
EUROPEAN_LANGS = {"Spanish", "French", "German", "Italian", "Portuguese",
                  "Dutch", "Polish", "Romanian", "Czech", "Greek", "Swedish",
                  "Russian", "Finnish", "Hungarian", "Danish", "Norwegian",
                  "Bulgarian", "Croatian", "Slovak", "Catalan", "Ukrainian"}


def lang_group(l):
    if l == "English":
        return "English"
    if l in INDIAN_VERNACULARS:
        return "Indian vernaculars"
    if l in EUROPEAN_LANGS:
        return "European"
    return "Others"


def clean(s):
    """Field text or empty string for null/placeholder values."""
    if pd.isna(s):
        return ""
    t = str(s).strip()
    return "" if t.lower() in {"", "nan", "none", "n/a", "na", "-"} else t


def combine(base, extra, sep=" "):
    """Append extra field to base; fall back to base when extra is empty."""
    return (base + sep + extra).strip() if extra else base


# ── metrics ──────────────────────────────────────────────────────────────────
def ranks_from_scores(S):
    """Rank (1-based) of the diagonal ground-truth passage per query,
    pessimistic within a tie group — identical to the benchmark pipeline:
    rank = 1 + (# strictly greater) + (# equal, excluding the target)."""
    N = S.shape[0]
    diag = S[np.arange(N), np.arange(N)]
    greater = (S > diag[:, None]).sum(axis=1)
    ties = (S == diag[:, None]).sum(axis=1)  # includes self
    return (greater + (ties - 1) + 1).astype(np.float64)


def rank_metrics(ranks):
    rr = 1.0 / ranks
    return {
        "MRR": float(rr.mean()),
        "Recall@1": float((ranks <= 1).mean()),
        "Recall@5": float((ranks <= 5).mean()),
        "Recall@10": float((ranks <= 10).mean()),
    }


def paired_bootstrap_p(a, b, B=BOOT_B, seed=SEED):
    """Two-sided paired bootstrap p-value on per-query values, H0-centred."""
    rng = np.random.default_rng(seed)
    d = a - b
    obs = d.mean()
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1) - obs        # centre at H0
    p = (np.abs(boot) >= abs(obs)).sum()
    return (p + 1) / (B + 1)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, oi in enumerate(order):
        running = max(running, (m - i) * pvals[oi])
        adj[oi] = min(1.0, running)
    return adj


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="Path to GeoQueryCA CSV (auto-detected if omitted)")
    ap.add_argument("--outdir", default=".")
    args, _ = ap.parse_known_args()   # notebook-safe

    # locate CSV
    csv_path = args.csv
    if csv_path is None:
        cands = [f for f in os.listdir(".") if f.lower().endswith(".csv")
                 and "geoquery" in f.lower()]
        if not cands:
            sys.exit("Pass --csv path/to/GeoQueryCA.csv")
        csv_path = cands[0]
        print(f"[auto] using {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.rename(columns={o: n for o, n in COL_ALIASES.items()
                            if o in df.columns and n not in df.columns})

    needed = ["Contextual_Query", "Cultural_Significance"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}. "
                 f"Found: {list(df.columns)}")

    df = df.dropna(subset=needed).reset_index(drop=True)
    N = len(df)
    print(f"Records: {N}")

    q_base = df["Contextual_Query"].map(clean).tolist()
    p_base = df["Cultural_Significance"].map(clean).tolist()
    cues = (df["Disambiguation_Cues"].map(clean).tolist()
            if "Disambiguation_Cues" in df.columns else [""] * N)
    near = (df["Nearby_Attractions"].map(clean).tolist()
            if "Nearby_Attractions" in df.columns else [""] * N)

    cov_cues = sum(bool(c) for c in cues) / N
    cov_near = sum(bool(x) for x in near) / N
    print(f"Field coverage — Disambiguation_Cues: {cov_cues:.1%}, "
          f"Nearby_Attractions: {cov_near:.1%}")

    groups = (df["Query_Language"].map(lang_group)
              if "Query_Language" in df.columns
              else pd.Series(["?"] * N))
    iv_mask = (groups == "Indian vernaculars").to_numpy()
    print(f"Indian vernacular queries: {iv_mask.sum()}")

    # text variants
    q_cues = [combine(q, c, ". Cues: ") for q, c in zip(q_base, cues)]
    q_only_cues = [c if c else q for q, c in zip(q_base, cues)]
    p_near = [combine(p, x, " Nearby: ") for p, x in zip(p_base, near)]
    p_only_near = [x if x else p for p, x in zip(p_base, near)]

    CONFIGS = {
        "baseline":    ("q_base", "p_base"),
        "+cues":       ("q_cues", "p_base"),
        "cues-only":   ("q_only_cues", "p_base"),
        "+nearby":     ("q_base", "p_near"),
        "nearby-only": ("q_base", "p_only_near"),
        "full":        ("q_cues", "p_near"),
    }
    TEXTS = {"q_base": q_base, "q_cues": q_cues, "q_only_cues": q_only_cues,
             "p_base": p_base, "p_near": p_near, "p_only_near": p_only_near}

    # encode each unique text variant once
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding with {MODEL_NAME} on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    EMB = {}
    for key, texts in TEXTS.items():
        prefix = PREFIX_Q if key.startswith("q") else PREFIX_P
        EMB[key] = model.encode([prefix + t for t in texts],
                                batch_size=BATCH, convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=True)

    # evaluate configurations
    results, per_query_rr = {}, {}
    for name, (qk, pk) in CONFIGS.items():
        S = EMB[qk] @ EMB[pk].T
        ranks = ranks_from_scores(S)
        rr = 1.0 / ranks
        per_query_rr[name] = rr
        m = rank_metrics(ranks)
        m_iv = rank_metrics(ranks[iv_mask]) if iv_mask.any() else {}
        results[name] = {"overall": m, "indian_vernaculars": m_iv}
        print(f"{name:12s} MRR={m['MRR']:.4f} R@1={m['Recall@1']:.4f} "
              f"R@5={m['Recall@5']:.4f}"
              + (f" | IV MRR={m_iv['MRR']:.4f}" if m_iv else ""))

    # significance vs baseline (Holm over the 5 ablations)
    base_rr = per_query_rr["baseline"]
    abl_names = [n for n in CONFIGS if n != "baseline"]
    p_raw = np.array([paired_bootstrap_p(per_query_rr[n], base_rr)
                      for n in abl_names])
    p_adj = holm(p_raw)
    sig = []
    for n, pr, pa in zip(abl_names, p_raw, p_adj):
        d = float(per_query_rr[n].mean() - base_rr.mean())
        sig.append({"config": n, "dMRR_vs_baseline": round(d, 4),
                    "p_raw": round(float(pr), 4),
                    "p_holm": round(float(pa), 4),
                    "significant": bool(pa < 0.05)})
        print(f"{n:12s} dMRR={d:+.4f} p_raw={pr:.4f} p_holm={pa:.4f} "
              f"{'SIG' if pa < 0.05 else 'n.s.'}")

    out = {"model": MODEL_NAME, "N": N,
           "coverage": {"Disambiguation_Cues": round(cov_cues, 4),
                        "Nearby_Attractions": round(cov_near, 4)},
           "results": results, "significance_vs_baseline": sig}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "ablation_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # LaTeX table
    def star(name):
        for s in sig:
            if s["config"] == name:
                return r"$^{*}$" if s["significant"] else ""
        return ""

    rows = []
    for name in CONFIGS:
        o = results[name]["overall"]
        iv = results[name]["indian_vernaculars"]
        rows.append(
            f"{name}{star(name)} & {o['MRR']:.4f} & {o['Recall@1']:.4f} & "
            f"{o['Recall@5']:.4f} & {o['Recall@10']:.4f} & "
            f"{iv.get('MRR', float('nan')):.4f} & "
            f"{iv.get('Recall@1', float('nan')):.4f} \\\\")

    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Field ablation with mE5-large (zero-shot, $N = "
        + f"{N:,}"
        + r"$). Query- and passage-side field combinations; IV $=$ Indian "
        r"vernacular subgroup. $^{*}$: MRR difference vs.\ baseline "
        r"significant at Holm-corrected $p<0.05$ (paired bootstrap, "
        r"$B{=}1{,}000$).}",
        r"\label{tab:field-ablation}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Configuration & MRR & R@1 & R@5 & R@10 & IV MRR & IV R@1 \\",
        r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(os.path.join(args.outdir, "tableG3_field_ablation.tex"),
              "w") as f:
        f.write(tex + "\n")

    # numbers file for prose
    md = ["# Field ablation numbers (Section 6.7)", "",
          f"Model: {MODEL_NAME}, N={N}",
          f"Coverage: cues {cov_cues:.1%}, nearby {cov_near:.1%}", ""]
    for name in CONFIGS:
        o = results[name]["overall"]
        iv = results[name]["indian_vernaculars"]
        md.append(f"- **{name}**: MRR={o['MRR']:.4f} R@1={o['Recall@1']:.4f} "
                  f"R@5={o['Recall@5']:.4f} R@10={o['Recall@10']:.4f}"
                  + (f" | IV: MRR={iv['MRR']:.4f} R@1={iv['Recall@1']:.4f}"
                     if iv else ""))
    md.append("")
    for s in sig:
        md.append(f"- {s['config']} vs baseline: dMRR="
                  f"{s['dMRR_vs_baseline']:+.4f}, p_holm={s['p_holm']:.4f} "
                  f"[{'SIG' if s['significant'] else 'n.s.'}]")
    with open(os.path.join(args.outdir, "ablation_numbers.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print("\nWrote ablation_results.json, tableG3_field_ablation.tex, "
          "ablation_numbers.md")


if __name__ == "__main__":
    main()