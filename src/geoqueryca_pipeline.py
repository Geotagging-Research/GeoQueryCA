"""
GeoQueryCA — End-to-End Pipeline
================================
Single script. Every value computed from the CSV.

  PHASE 1 — Dataset profiling
      Data quality, diversity (entropy/Gini), geographic, language, category,
      and confidence-score statistics; six descriptive figures; JSON + MD.

  PHASE 2 — Zero-shot retrieval benchmark (publication-grade)
      Six models (BM25, mBERT, XLM-R, LaBSE, mE5-large, BGE-M3).
      Six metrics (MRR, nDCG@{5,10}, Recall@{1,5,10}) with bootstrap 95% CIs.
      Paired-bootstrap significance tests + Holm–Bonferroni correction.
      Cross-lingual breakdown (in-pool & global) and per-language metrics.
      Publication tables (3/4/5), figures (forest, heatmap, sig-matrix,
      MRR bar), booktabs LaTeX exports, methodology MD.

Outputs (all under ./geoqueryca_outputs/):
    descriptive/   — Phase 1: fig1..fig6, evaluation_metrics.json,
                     evaluation_report.md
    benchmark/     — Phase 2: tables 3/4/5, figures, *.tex, metrics.json,
                     overall_metrics.csv, per_query_ranks.npz, METHODOLOGY.md
    cache/         — embedding cache (re-runs are fast)
    report.md      — unified summary linking everything

Colab-friendly. GPU strongly recommended for Phase 2.
Toggle phases via CONFIG['run_profiling'] / CONFIG['run_benchmark'].
"""

# ════════════════════════════════════════════════════════════════════════════
# 0. Dependency bootstrap  (one-time, idempotent, cached)
# ════════════════════════════════════════════════════════════════════════════
# WHY THIS IS FAST ON RE-RUNS
# ───────────────────────────
# Every slow operation is guarded by a runtime-level flag stored in the
# interpreter's own module dict (__builtins__ is not used; we store in a
# private module so the flag survives cell re-runs within the same session).
#
# _DEPS_DONE   — set after all packages are confirmed present.
#                On re-run: skips the entire import-check loop (~0.3 s saved).
# _TV_DONE     — set after the torchvision subprocess check completes.
#                On re-run: skips the subprocess entirely (~1–4 s saved on
#                machines where torchvision IS installed but healthy).
#
# Cold start (fresh runtime): behaves identically to the original — installs
# missing packages, runs the subprocess check, then sets both flags.
# ─────────────────────────────────────────────────────────────────────────────
import sys, subprocess, importlib, types as _types

# Use a throw-away module as a cross-cell flag store (survives cell re-runs).
if "_geoqueryca_bootstrap" not in sys.modules:
    sys.modules["_geoqueryca_bootstrap"] = _types.ModuleType("_geoqueryca_bootstrap")
_bs = sys.modules["_geoqueryca_bootstrap"]


# ── 0a. Package installer ────────────────────────────────────────────────────
def _pip_install(args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + args)

def _torch_pin():
    """Lock pip to the already-installed torch version so later installs
    can't swap Colab's CUDA-matched build (root cause of CUBLAS errors)."""
    try:
        import torch
        return [f"torch=={torch.__version__.split('+')[0]}"]
    except ImportError:
        return []

def _ensure(pkg, imp=None):
    """Import-test first; only call pip if the package is truly absent."""
    mod_name = imp or pkg
    for attempt in range(2):
        try:
            importlib.import_module(mod_name)
            return
        except ImportError:
            if attempt == 0:
                print(f"  installing {pkg} …")
                _pip_install([pkg] + _torch_pin())
        except AttributeError as e:
            # Circular-import partial-init — purge stale cache and retry once
            if "partially initialized module" in str(e) or \
               "has no attribute" in str(e):
                if attempt == 0:
                    for _k in [k for k in sys.modules
                                if k == mod_name or k.startswith(mod_name + ".")]:
                        del sys.modules[_k]
                    importlib.invalidate_caches()
                else:
                    print(f"  ⚠  import issue for {pkg}: {e}")
                    return
            else:
                raise

# Only run the full check loop once per Colab session.
if not getattr(_bs, "_DEPS_DONE", False):
    print("Checking dependencies …")
    DEPENDENCIES = [
        ("pandas",                None),
        ("numpy",                 None),
        ("scipy",                 None),
        ("tqdm",                  None),
        ("matplotlib",            None),
        ("seaborn",               None),
        ("pycountry",             "pycountry"),
        ("pycountry-convert",     "pycountry_convert"),
        ("torch",                 None),   # first — never let pip replace it
        ("rank-bm25",             "rank_bm25"),
        ("transformers",          None),
        ("sentence-transformers", "sentence_transformers"),
        ("scikit-learn",          "sklearn"),
        ("requests",              None),   # Wikipedia / Wikidata API calls
    ]
    for _pkg, _imp in DEPENDENCIES:
        _ensure(_pkg, _imp)
    _bs._DEPS_DONE = True
    print("Dependencies OK.")
else:
    print("Dependencies already verified — skipping.")


# ── 0b. Core imports ─────────────────────────────────────────────────────────
import os, json, math, re, random, hashlib, warnings, textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pycountry
import pycountry_convert as pc
from scipy.stats import entropy
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")


# ── 0c. torch / torchvision mismatch guard ───────────────────────────────────
# Problem : transformers lazily imports torchvision; a version mismatch causes
#           "operator torchvision::nms does not exist" deep inside model loading.
# Fix     : test in an ISOLATED subprocess so we don't pollute this process.
#           If broken, uninstall torchvision before transformers is ever loaded.
# Speed   : the subprocess takes 1–4 s even when healthy, so we skip it on
#           re-runs by checking _TV_DONE (set once per session).
def _guard_torchvision():
    import importlib.util
    if importlib.util.find_spec("torchvision") is None:
        return  # not installed → transformers skips it automatically

    # Already checked this session — don't pay the subprocess cost again.
    if getattr(_bs, "_TV_DONE", False):
        return

    test = subprocess.run(
        [sys.executable, "-c",
         "import torch, torchvision; torch.ops.torchvision.nms"],
        capture_output=True, timeout=30)

    if test.returncode != 0:
        print("⚠  torch/torchvision mismatch — removing torchvision so "
              "transformers imports cleanly …")
        subprocess.call([sys.executable, "-m", "pip", "uninstall",
                         "-y", "-q", "torchvision"])
        for _m in [m for m in sys.modules if m.startswith("torchvision")]:
            del sys.modules[_m]
        importlib.invalidate_caches()
        print("   ✓ torchvision removed — text model loading will now work.")
    else:
        print("torchvision OK.")

    _bs._TV_DONE = True   # skip subprocess on every subsequent re-run

_guard_torchvision()

# ════════════════════════════════════════════════════════════════════════════
# 1. Configuration (single source of truth)
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # phase toggles
    "run_profiling":        True,
    "run_benchmark":        True,
    "run_finetuning":       True,
    "run_coord_validation": True,   # Phase 1.5: validate coords vs Wikipedia/Wikidata/Google
    # ── Google Maps Places API ─────────────────────────────────────────────
    # Get a free key at https://console.cloud.google.com/
    # Enable: Places API + Geocoding API
    # Free tier: 200 USD/month credit (~40,000 Places requests)
    # Set to None to skip Google Maps validation layer
    "google_maps_api_key":  None,   # ← paste your key here: "AIza..."
    # Google Maps validation settings
    "gmaps_max_accept_km":  50,     # reject Google result if >50 km from dataset
    "gmaps_use_geocoding":  True,   # also try Geocoding API (address→coords)
    "gmaps_use_places":     True,   # use Places Text Search API
    # reproducibility
    "random_seed":          42,
    # benchmark
    "bootstrap_B":          1000,
    "bootstrap_alpha":      0.05,
    "min_language_n":       50,          # threshold for per-language reporting
    "batch_size":           32,
    "max_seq_length":       256,
    "k_values":             (1, 5, 10),
    "ndcg_k_values":        (5, 10),
    "smoke_test_n":         None,        # set int (e.g. 300) for quick check
    # fine-tuning (Phase 3)
    "finetune_base_model":  "bert-base-multilingual-cased",
    "finetune_epochs":      3,
    "finetune_batch_size":  16,
    "finetune_warmup_steps":100,
    "finetune_test_size":   0.30,
    # paths
    "root_dir":             "geoqueryca_outputs",
    "models": [
        {"label": "BM25",     "kind": "bm25",  "name": "BM25"},
        {"label": "mBERT",    "kind": "hf",    "name": "bert-base-multilingual-cased"},
        {"label": "XLM-R",    "kind": "hf",    "name": "xlm-roberta-base"},
        {"label": "LaBSE",    "kind": "sbert", "name": "sentence-transformers/LaBSE"},
        # Modern dense retrievers (2024 SOTA cross-lingual)
        {"label": "mE5-large","kind": "sbert", "name": "intfloat/multilingual-e5-large",
         "prefix_q": "query: ", "prefix_p": "passage: "},
        {"label": "BGE-M3",   "kind": "sbert", "name": "BAAI/bge-m3"},
        # NOTE: mGTE-base (Alibaba-NLP/gte-multilingual-base) removed. Its custom
        # remote modeling code triggers a CUDA device-side assert that corrupts
        # the CUDA context for the rest of the process, causing later phases to
        # crash. The paper reports six models; this keeps code and paper aligned.
    ],
}

ROOT = CONFIG["root_dir"]
DESC_DIR  = os.path.join(ROOT, "descriptive")
BENCH_DIR = os.path.join(ROOT, "benchmark")
CACHE_DIR = os.path.join(ROOT, "cache")
for d in (ROOT, DESC_DIR, BENCH_DIR, CACHE_DIR): os.makedirs(d, exist_ok=True)

def _seed_everything(s):
    """Seed all RNGs. Gracefully skips CUDA seeding if the GPU state is
    corrupted (AcceleratorError / device-side assert) and falls back to CPU
    instead of crashing the whole CONFIG cell."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception as _seed_err:
            print(f"⚠  CUDA seed failed ({type(_seed_err).__name__}) — "
                  "GPU state is corrupted.")
            print("   Fix: Runtime → Restart session, then re-run from the top.")
            print("   Continuing on CPU for this session.\n")
            return False   # signals caller to fall back to CPU
    return True

# Attempt seeding; if CUDA is broken, force CPU for the whole session
_cuda_ok = _seed_everything(CONFIG["random_seed"])

DEVICE = torch.device(
    "cuda" if (torch.cuda.is_available() and _cuda_ok) else "cpu")
print(f"Device : {DEVICE}    |    seed : {CONFIG['random_seed']}")
print(f"torch  : {torch.__version__}  (built for CUDA {torch.version.cuda})")

# ── GPU self-test ─────────────────────────────────────────────────────────────
# Catches CUBLAS_STATUS_NOT_SUPPORTED, AcceleratorError, device-side asserts,
# and OOM — all of which mean the GPU/torch stack is unusable this session.
# Root cause is almost always a pip install that swapped Colab's CUDA torch build.
if DEVICE.type == "cuda":
    try:
        _t = torch.randn(64, 64, device=DEVICE)
        _ = (_t @ _t).sum().item()
        del _t
        torch.cuda.empty_cache()
        print("GPU self-test: ✓ matmul OK")
    except Exception as _e:
        print("\n" + "!" * 72)
        print("GPU self-test FAILED:", str(_e).splitlines()[0][:160])
        print("!" * 72)
        print("\nRoot cause: corrupted CUDA state or torch/driver version mismatch.")
        print("\nFIX — do ONE of these, then re-run from Cell 1:")
        print("  • Runtime → Restart session          (fixes most cases)")
        print("  • Runtime → Disconnect and delete runtime → reconnect to fresh GPU")
        print("  • Do NOT pip-install/upgrade torch — that breaks the CUDA build.")
        print("\nFalling back to CPU so the rest of the pipeline still runs.\n")
        DEVICE = torch.device("cpu")
        torch.cuda.empty_cache()

if CONFIG["run_benchmark"] and DEVICE.type == "cpu":
    print("  ⚠  Phase 2 will run on CPU (~10–30 min/model).")
    print("     For GPU: Runtime → Change runtime type → T4 GPU, restart, re-run.")

# ════════════════════════════════════════════════════════════════════════════
# 2. Shared utilities — load + clean CSV
# ════════════════════════════════════════════════════════════════════════════
def _find_csv():
    """Locate the GeoQueryCA CSV.

    Priority order:
      1. Google Colab file-upload widget  (interactive, user picks the file)
      2. Already-uploaded file in the Colab runtime  (re-run without re-uploading)
      3. Local fallback paths  (running outside Colab)
    """
    # ── 1. Try Colab upload widget ────────────────────────────────────────
    _in_colab = False
    try:
        import google.colab          # noqa – just testing availability
        _in_colab = True
    except ImportError:
        pass

    if _in_colab:
        from google.colab import files as _colab_files

        # Check whether the file was already uploaded in a previous cell/run
        # so the user is not forced to re-upload on every execution.
        _KNOWN_NAMES = [
            "GEOQUERYCA_DATASET_V1.0.csv",
            "data/GEOQUERYCA_DATASET_V1.0.csv",
            "GEOQUERYCA_JUNE.csv",
            "GEOQUERYCA_DATASET_Publish.csv",
        ]
        for _name in _KNOWN_NAMES:
            if os.path.exists(_name):
                print(f"[Colab] Found previously uploaded file: {_name}")
                return _name

        # File not yet present → open the upload widget
        print("=" * 60)
        print("UPLOAD REQUIRED")
        print("Please upload your GeoQueryCA CSV file using the")
        print("file-picker that appears below.")
        print("Accepted filenames: GEOQUERYCA_JUNE.csv  or")
        print("                    GEOQUERYCA_DATASET_Publish.csv")
        print("=" * 60)
        uploaded = _colab_files.upload()   # blocks until user selects a file

        if not uploaded:
            raise FileNotFoundError(
                "No file was uploaded. Re-run this cell and select "
                "your GeoQueryCA CSV in the file-picker.")

        fname = list(uploaded.keys())[0]

        # Rename to a canonical name so re-runs skip the widget
        canonical = "GEOQUERYCA_JUNE.csv"
        if fname != canonical:
            import shutil
            shutil.copy(fname, canonical)
            print(f"[Colab] Saved as '{canonical}' for future re-runs.")
            fname = canonical

        print(f"[Colab] Loaded: {fname}  ({len(uploaded[list(uploaded.keys())[0]]):,} bytes)")
        return fname

    # ── 2. Local / non-Colab fallback ────────────────────────────────────
    for p in [
        "data/GEOQUERYCA_DATASET_V1.0.csv",        # public release layout
        "../data/GEOQUERYCA_DATASET_V1.0.csv",     # when run from src/
        "GEOQUERYCA_DATASET_V1.0.csv",
        "GEOQUERYCA_JUNE.csv",
        "GEOQUERYCA_DATASET_Publish.csv",
        "/mnt/user-data/uploads/GEOQUERYCA_JUNE.csv",
        "/mnt/user-data/uploads/GEOQUERYCA_DATASET_Publish.csv",
    ]:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "CSV not found. Options:\n"
        "  • In Colab  : run this script and use the upload widget.\n"
        "  • Locally   : run from the repository root (data/GEOQUERYCA_DATASET_V1.0.csv)."
    )

CSV_PATH = _find_csv()

# ── Encoding + delimiter + parser-robust CSV loader ───────────────────────────
import io as _io, csv as _csv

def _sniff_delimiter(raw_bytes):
    """Determine delimiter by counting consistent field counts across rows.
    Strips UTF-8 BOM before analysis. Tries comma, tab, semicolon, pipe.
    Returns the delimiter that gives the most consistent and largest column count."""
    sample_bytes = raw_bytes[:16384]
    if sample_bytes.startswith(b"\xef\xbb\xbf"):
        sample_bytes = sample_bytes[3:]

    text = sample_bytes.decode("utf-8", errors="replace")
    lines = [l for l in text.split("\n") if l.strip()][:20]  # first 20 rows

    best_sep, best_score = ",", 0
    for sep in [",", "\t", ";", "|"]:
        counts = [len(l.split(sep)) for l in lines]
        if not counts:
            continue
        max_count = max(counts)
        if max_count < 3:          # must produce ≥3 columns
            continue
        # Score = column count × consistency (fraction of rows with max count)
        consistency = sum(1 for c in counts if c == max_count) / len(counts)
        score = max_count * consistency
        if score > best_score:
            best_score = score
            best_sep = sep
    return best_sep

def _load_csv_robust(path):
    """Load CSV handling UTF-8 BOM, encoding variants, trailing empty cols,
    and both comma-separated and tab-separated formats.
    Returns (DataFrame, encoding_used, sep_used)."""
    with open(path, "rb") as _f:
        raw = _f.read()

    # Determine BOM and strip it
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    payload  = raw[3:] if has_bom else raw

    # Detect encoding: UTF-8 if BOM present, otherwise try latin-1 / cp1252
    enc_used = None
    text     = None
    _enc_order = (["utf-8"] if has_bom else []) + \
                 ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    for enc in _enc_order:
        try:
            text = payload.decode(enc)
            enc_used = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text     = payload.decode("latin-1", errors="replace")
        enc_used = "latin-1(lossy)"

    sep = _sniff_delimiter(raw)

    # Parse — python engine tolerates ragged rows (trailing empty cols)
    df = pd.read_csv(_io.StringIO(text), sep=sep,
                     engine="python", on_bad_lines="skip")

    # Drop trailing all-NaN / unnamed columns (common Excel artefact)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")

    return df, enc_used, sep

df_raw, _used_enc, _used_sep = _load_csv_robust(CSV_PATH)

if df_raw is None:
    raise RuntimeError(
        "Could not load the CSV with any encoding, delimiter, or parser "
        "strategy.\nPlease re-save from Excel as: "
        "File → Save As → CSV UTF-8 (comma delimited).")

# If file was TSV but saved with .csv extension, normalise to comma-sep in mem
# (no-op if already comma-separated)
if df_raw is None or df_raw.shape[1] < 3:
    raise RuntimeError(
        "Could not load the CSV correctly. "
        "Please re-save from Excel as: File → Save As → "
        "CSV UTF-8 (comma delimited).")

print(f"Loaded {CSV_PATH}  "
      f"[encoding={_used_enc}, sep={repr(_used_sep)}]  "
      f"shape: {df_raw.shape}")

# ── Column name normalisation ─────────────────────────────────────────────────
# Different versions of the CSV use different column names for the same fields.
# Normalise to the canonical names expected by the rest of the pipeline.
_COL_ALIASES = {
    # The public v1.0 release names the landmark column "Landmark". It must map
    # to "Monument" so that the Phase-3 train/test split groups records by
    # landmark (as in the paper: 1,407 train / 591 test). Without this alias the
    # split silently falls back to grouping by passage text (1,404 / 597).
    "Landmark":           "Monument",
    "landmark":           "Monument",
    "Point_of_Interest":  "Monument",
    "point_of_interest":  "Monument",
    "POI":                "Monument",
    "City/Region":        "City",
    "city/region":        "City",
    "CityRegion":         "City",
    "Lat":                "Latitude",
    "lat":                "Latitude",
    "Lon":                "Longitude",
    "Long":               "Longitude",
    "lon":                "Longitude",
    "long":               "Longitude",
    "Cultural_Sig":       "Cultural_Significance",
    "Query":              "Contextual_Query",
    "Language":           "Query_Language",
}
_renamed = {old: new for old, new in _COL_ALIASES.items()
            if old in df_raw.columns and new not in df_raw.columns}
if _renamed:
    df_raw = df_raw.rename(columns=_renamed)
    print(f"  Column aliases applied: {_renamed}")
df_raw = df_raw.loc[:, ~df_raw.columns.str.match(r"^Unnamed")]
print(f"\nLoaded {CSV_PATH}  — raw shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")

# Strip whitespace on all object columns
for c in df_raw.select_dtypes(include=["object", "string"]).columns:
    df_raw[c] = df_raw[c].astype("string").str.strip()

# Country normalisation (resolves aliases of names already present in the CSV)
COUNTRY_ALIASES = {
    "USA": "United States", "U.S.A.": "United States", "US": "United States",
    "UK": "United Kingdom", "U.K.": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Russia": "Russian Federation",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
    "Vietnam": "Viet Nam", "Iran": "Iran, Islamic Republic of",
    "Czech Republic": "Czechia", "Burma": "Myanmar",
    "Ivory Coast": "Côte d'Ivoire",
}
if "Country" in df_raw.columns:
    df_raw["Country"] = df_raw["Country"].replace(COUNTRY_ALIASES)

# Confidence numeric
if "Confidence_Score" in df_raw.columns:
    df_raw["Confidence_Score"] = pd.to_numeric(df_raw["Confidence_Score"], errors="coerce")

# ── DMS → Decimal Degrees conversion ────────────────────────────────────────
# Converts Degrees-Minutes-Seconds strings like "31°37′12″ N" to decimal float.
# Works for both original DMS columns AND any pre-existing decimal columns so the
# pipeline is robust regardless of which version of the CSV is loaded.
def _dms_to_decimal(dms_str):
    """Parse DMS string → float decimal degrees.

    Handles two formats:
      Standard  : "31°37′12″ N"  (U+00B0, U+2032, U+2033)
      Lossy     : "31°37?12? N"  (° preserved as 0xB0, ′ and ″ mangled to ?)
                  — occurs when UTF-8 multibyte chars are saved in latin-1
    Returns NaN on parse failure.
    """
    if pd.isna(dms_str):
        return float("nan")
    s = str(dms_str).strip()
    try:
        return float(s)
    except ValueError:
        pass

    _N = r"(\d+(?:\.\d+)?)"

    def _extract(g):
        """Extract (direction, deg, min, sec) from match groups."""
        g = [x for x in g if x is not None]
        if g[0].upper() in "NSEW":
            return g[0], g[1], g[2], g[3]
        return g[3], g[0], g[1], g[2]

    def _calc(direction, deg, mn, sec):
        d = float(deg) + float(mn)/60.0 + float(sec)/3600.0
        return round(-d if direction.upper() in "SW" else d, 6)

    # ── Pattern A: lossy encoding — applied FIRST on the RAW string ────────
    # Matches "31°37?12? N" where ° is intact but ′ and ″ became ?
    pat_lossy = (rf"{_N}[°º\xb0\xba]\s*{_N}\?\s*{_N}\?\s*([NSEWnsew])"
                 rf"|([NSEWnsew])\s*{_N}[°º\xb0\xba]\s*{_N}\?\s*{_N}\?")
    m = re.search(pat_lossy, s)
    if m:
        return _calc(*_extract(m.groups()))

    # ── Pattern B: standard Unicode — normalise then match ─────────────────
    s2 = (s
          .replace("\u00b0","d").replace("\u00ba","d")
          .replace("\u2032","m").replace("\u0027","m").replace("\u2019","m")
          .replace("\u2033","s").replace("\u0022","s").replace("\u201d","s")
          )
    pat_std = (rf"{_N}d\s*{_N}m\s*{_N}s\s*([NSEWnsew])"
               rf"|([NSEWnsew])\s*{_N}d\s*{_N}m\s*{_N}s")
    m2 = re.search(pat_std, s2)
    if m2:
        return _calc(*_extract(m2.groups()))

    return float("nan")

_LAT_COLS = [c for c in df_raw.columns if c.lower() in ("latitude",  "lat")]
_LON_COLS = [c for c in df_raw.columns if c.lower() in ("longitude", "lon", "long")]

# Always create canonical Lat_DD / Lon_DD columns from whichever source exists
if _LAT_COLS:
    df_raw["Lat_DD"] = df_raw[_LAT_COLS[0]].apply(_dms_to_decimal)
else:
    df_raw["Lat_DD"] = float("nan")

if _LON_COLS:
    df_raw["Lon_DD"] = df_raw[_LON_COLS[0]].apply(_dms_to_decimal)
else:
    df_raw["Lon_DD"] = float("nan")

_n_ok_lat = df_raw["Lat_DD"].notna().sum()
_n_ok_lon = df_raw["Lon_DD"].notna().sum()
print(f"Coordinate conversion → Lat_DD: {_n_ok_lat}/{len(df_raw)} OK  |  "
      f"Lon_DD: {_n_ok_lon}/{len(df_raw)} OK")
if _n_ok_lat < len(df_raw) * 0.9:
    print("  ⚠  >10% of latitude values failed to parse — check source column format.")

# ── Haversine distance helper (vectorised, km) ────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Element-wise haversine distance in kilometres.
    All inputs are numpy arrays or scalars in decimal degrees."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Drop rows missing the essentials BOTH phases need
ESSENTIAL = [c for c in
             ["Country", "Query_Language", "Category",
              "Contextual_Query", "Cultural_Significance"]
             if c in df_raw.columns]
df = df_raw.dropna(subset=ESSENTIAL).copy()
for c in ESSENTIAL:
    df = df[df[c].astype(str) != ""]
df = df.drop_duplicates(
        subset=[c for c in ("Contextual_Query","Cultural_Significance")
                if c in df.columns])
df = df.sort_values([c for c in ("Query_Language","Contextual_Query")
                     if c in df.columns]).reset_index(drop=True)

if CONFIG["smoke_test_n"]:
    df = df.sample(n=min(CONFIG["smoke_test_n"], len(df)),
                   random_state=CONFIG["random_seed"]).reset_index(drop=True)
    print(f"  [smoke test] subsampled to N = {len(df)}")

N = len(df)
print(f"\nClean evaluation set: N = {N:,} rows")

# Coordinate availability in the cleaned frame
HAS_COORDS = ("Lat_DD" in df.columns and "Lon_DD" in df.columns
              and df["Lat_DD"].notna().mean() > 0.5
              and df["Lon_DD"].notna().mean() > 0.5)
if HAS_COORDS:
    _coord_ok = (df["Lat_DD"].notna() & df["Lon_DD"].notna()).sum()
    print(f"  Coordinates available for geo-error metrics: {_coord_ok:,}/{N:,} rows")
    # Precompute arrays used by every geo-error function
    LAT_ARR = df["Lat_DD"].to_numpy(dtype=np.float64)
    LON_ARR = df["Lon_DD"].to_numpy(dtype=np.float64)
else:
    print("  ⚠  Coordinate columns missing/sparse — geo-error metrics will be skipped.")

# ════════════════════════════════════════════════════════════════════════════
# 3. Geographic + language grouping  (used by both phases)
# ════════════════════════════════════════════════════════════════════════════
_CONT_MAP = {"AF":"Africa", "AS":"Asia", "EU":"Europe",
             "NA":"Americas", "SA":"Americas", "OC":"Oceania", "AN":"Antarctica"}
def country_to_continent(name):
    try:
        c = pycountry.countries.lookup(str(name))
        return _CONT_MAP.get(pc.country_alpha2_to_continent_code(c.alpha_2), "Other")
    except Exception:
        return "Other"

if "Region" not in df.columns or df["Region"].isna().mean() > 0.5:
    df["Region"] = df["Country"].apply(country_to_continent)
else:
    miss = df["Region"].isna()
    df.loc[miss, "Region"] = df.loc[miss, "Country"].apply(country_to_continent)

INDIAN_VERNACULARS = {"Hindi","Marathi","Bengali","Tamil","Telugu","Kannada",
                      "Malayalam","Odia","Oriya","Punjabi","Gujarati","Assamese",
                      "Urdu","Sanskrit","Sindhi","Konkani","Maithili","Nepali",
                      "Sinhala"}
EUROPEAN_LANGS = {"Spanish","French","German","Italian","Portuguese","Dutch",
                  "Polish","Romanian","Czech","Greek","Swedish","Russian",
                  "Finnish","Hungarian","Danish","Norwegian","Bulgarian",
                  "Croatian","Slovak","Catalan","Ukrainian"}
GROUPS_ORDER = ["English","Indian vernaculars","European","Others"]

def lang_group(l):
    if l == "English":            return "English"
    if l in INDIAN_VERNACULARS:   return "Indian vernaculars"
    if l in EUROPEAN_LANGS:       return "European"
    return "Others"

df["group"] = df["Query_Language"].apply(lang_group)

# ════════════════════════════════════════════════════════════════════════════
# Shared figure styling
# ════════════════════════════════════════════════════════════════════════════
# ── Springer Nature / Q1 Journal Style Guide ─────────────────────────────────
# Column widths (inches):  single = 3.54 in  |  1.5-col = 5.51 in  |  double = 7.48 in
# Max page height          ~9.45 in  (A4 with margins)
# Minimum font size        6 pt (axis labels), 7 pt (body), 8 pt (captions)
# Resolution               ≥ 600 dpi line art / ≥ 300 dpi halftone (we use 600)
# Colour mode              CMYK-safe RGB (no neon / web-only colours)
# Font                     Helvetica-compatible sans-serif → Arial via matplotlib
#
# Colour palette — Nature Portfolio style (CMYK-safe, colour-blind accessible):
#   Primary blues  : #1F4E79 (deep navy), #2E75B6 (mid blue), #9DC3E6 (pale blue)
#   Accent greens  : #375623 (dark), #538135 (mid), #A9D18E (light)
#   Accent orange  : #843C0C (dark), #C55A11 (mid), #F4B183 (light)
#   Greys          : #404040, #7F7F7F, #D9D9D9
#   Warning red    : #C00000
#   Highlight      : #FFF2CC  (subtle yellow, used sparingly)
#
# All figure sizes obey the column-width rule; height chosen for ≤ 3:2 aspect.
# ─────────────────────────────────────────────────────────────────────────────

# ── Springer Nature column widths (inches) ───────────────────────────────────
SN_1COL   = 3.54    # single column
SN_15COL  = 5.51    # 1.5 column
SN_2COL   = 7.48    # full double column
SN_DPI    = 600     # raster export DPI (600 for line art, acceptable for halftone)

# ── CMYK-safe colour palette ─────────────────────────────────────────────────
# Blues (primary series) — 8 homogeneous shades, dark→light
C_NAVY    = "#1F3864"   # darkest blue  (was #1F4E79)
C_BLUE1   = "#2E5F9E"   # dark-mid blue (was #2E75B6)
C_BLUE2   = "#4472C4"   # mid blue
C_BLUE3   = "#5B8DB8"   # mid-light blue
C_LBLUE   = "#9DC3E6"   # light blue
C_VBLUE   = "#BDD7EE"   # very light blue
C_PVBLUE  = "#DEEAF1"   # pale blue (table alternating row)
# Greys (secondary — for contrast elements only)
C_DGREY   = "#404040"
C_MGREY   = "#7F7F7F"
C_LGREY   = "#D9D9D9"
C_WHITE   = "#FFFFFF"
# Semantic — kept minimal; only red for error/warning, yellow for highlight
C_RED     = "#C00000"
C_HILITE  = "#FFF2CC"   # highlight (best-cell in tables — used very sparingly)
C_REDBG   = "#FCE4D6"   # soft red-tint background for warning rows

# Ordered palette — 8 homogeneous blue/grey shades, colour-blind safe
# Used for ALL categorical series so every figure looks visually consistent
PALETTE = [C_NAVY, C_BLUE1, C_BLUE2, C_BLUE3,
           C_LBLUE, C_VBLUE, C_MGREY, C_LGREY]

# Legacy aliases — kept so no figure code breaks
BLUE_DARK  = C_NAVY
BLUE_MED   = C_BLUE1
BLUE_LIGHT = C_PVBLUE
BLUE_ALT   = "#EBF3FB"
WARN_BG    = C_REDBG
WARN_FG    = C_RED
TEXT_DARK  = C_DGREY
TEXT_LIGHT = C_WHITE
WHITE      = C_WHITE
HIGHLIGHT  = C_HILITE
# Kept for fine-tuned model rows (subtle teal-grey instead of green)
C_GRN      = "#4E7B8C"
C_DGRN     = "#2F4F5C"
C_LGRN     = "#A8C5CE"

# ── Matplotlib global rcParams (Springer Nature submission profile) ───────────
import matplotlib
matplotlib.rcParams.update({
    # Font — Arial is the closest free substitute for Helvetica
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            7,            # base size (pt)
    "axes.titlesize":       8,
    "axes.labelsize":       7,
    "xtick.labelsize":      6,
    "ytick.labelsize":      6,
    "legend.fontsize":      6,
    "legend.title_fontsize":7,
    # Weight
    "axes.titleweight":     "bold",
    "axes.labelweight":     "normal",
    # Lines & ticks
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "xtick.minor.size":     1.5,
    "ytick.minor.size":     1.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.linewidth":      1.0,
    "patch.linewidth":      0.5,
    # Colours
    "axes.edgecolor":       C_DGREY,
    "axes.labelcolor":      C_DGREY,
    "xtick.color":          C_DGREY,
    "ytick.color":          C_DGREY,
    "text.color":           C_DGREY,
    "axes.facecolor":       C_WHITE,
    "figure.facecolor":     C_WHITE,
    # Grid — minimal, light
    "axes.grid":            True,
    "grid.color":           C_LGREY,
    "grid.linewidth":       0.4,
    "grid.alpha":           0.6,
    "axes.axisbelow":       True,
    # Spines — top/right off (Nature style)
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # Legend
    "legend.frameon":       True,
    "legend.framealpha":    0.85,
    "legend.edgecolor":     C_LGREY,
    "legend.handlelength":  1.2,
    "legend.handletextpad": 0.4,
    "legend.borderpad":     0.4,
    # Figure / layout
    "figure.dpi":           150,          # screen preview; _save() uses SN_DPI
    "savefig.dpi":          SN_DPI,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    C_WHITE,
    "figure.constrained_layout.use": False,  # we call tight_layout() explicitly
})

# ════════════════════════════════════════════════════════════════════════════
# Model characteristics — paper-ready Table 2 metadata.  Lookup is by label
# in CONFIG["models"]. Add a new entry whenever you add a new model.  Param
# counts and embedding dims are taken from each model's official HF card or
# published paper (URLs in METHODOLOGY.md).
# ════════════════════════════════════════════════════════════════════════════
MODEL_METADATA = {
    "BM25":     {"params": "—",    "dim": "—",
                 "retrieval_trained": "No (lexical)",
                 "indian_support":    "Limited",
                 "base":              "Okapi BM25",
                 "year":              "1994"},
    "mBERT":    {"params": "178M", "dim": "768",
                 "retrieval_trained": "No (MLM + NSP)",
                 "indian_support":    "Yes (104 langs)",
                 "base":              "BERT-base",
                 "year":              "2018"},
    "XLM-R":    {"params": "270M", "dim": "768",
                 "retrieval_trained": "No (MLM)",
                 "indian_support":    "Yes (100 langs)",
                 "base":              "RoBERTa-base",
                 "year":              "2019"},
    "LaBSE":    {"params": "471M", "dim": "768",
                 "retrieval_trained": "Yes (dual-encoder, translation pairs)",
                 "indian_support":    "Yes (109 langs)",
                 "base":              "BERT-base + custom vocab",
                 "year":              "2020"},
    "mE5-base": {"params": "278M", "dim": "768",
                 "retrieval_trained": "Yes (contrastive, text pairs)",
                 "indian_support":    "Yes (100 langs)",
                 "base":              "XLM-R-base",
                 "year":              "2024"},
    "mE5-large":{"params": "560M", "dim": "1024",
                 "retrieval_trained": "Yes (contrastive, text pairs)",
                 "indian_support":    "Yes (100 langs)",
                 "base":              "XLM-R-large",
                 "year":              "2024"},
    "BGE-M3":   {"params": "568M", "dim": "1024",
                 "retrieval_trained": "Yes (multi-functional contrastive + distillation)",
                 "indian_support":    "Yes (100+ langs)",
                 "base":              "XLM-R-large",
                 "year":              "2024"},
    "mGTE-base":{"params": "305M", "dim": "768",
                 "retrieval_trained": "Yes (multi-stage contrastive, encoder-only)",
                 "indian_support":    "Yes (70+ langs)",
                 "base":              "Custom encoder (mGTE-base)",
                 "year":              "2024"},
    # Fine-tuned variants produced by Phase 3
    "FT-mBERT": {"params": "178M", "dim": "768",
                 "retrieval_trained": "Yes (fine-tuned, MNR loss on GeoQueryCA)",
                 "indian_support":    "Yes (104 langs)",
                 "base":              "mBERT-cased",
                 "year":              "this work"},
}

# tiny LaTeX-escaping helper — used by every table .tex export
def _tex_escape(s): return str(s).replace("&","\\&").replace("_","\\_")

# ════════════════════════════════════════════════════════════════════════════
# Table 2 renderer — reusable so Phase 2 can render zero-shot only, and
# Phase 3 can overwrite with the fine-tuned row appended.
# ════════════════════════════════════════════════════════════════════════════
def render_table2(labels, outdir, basename="table2_model_characteristics",
                  title_suffix=""):
    """Render Table 2 (Model Characteristics) as PNG/PDF + LaTeX.
       `labels` is a list of model labels — each must have an entry in
       MODEL_METADATA (missing entries fall back to em-dashes)."""
    t2_cols = ["Model", "Params", "Embedding Dim",
               "Retrieval-trained", "Indian Lang Support"]
    t2_data = []
    for lab in labels:
        meta = MODEL_METADATA.get(lab, {"params":"—","dim":"—",
                                        "retrieval_trained":"—",
                                        "indian_support":"—"})
        t2_data.append([lab, meta["params"], meta["dim"],
                        meta["retrieval_trained"], meta["indian_support"]])

    fig, ax = plt.subplots(figsize=(SN_2COL, 0.28 * len(labels) + 0.9))
    fig.patch.set_facecolor(WHITE); ax.axis("off")
    table = ax.table(cellText=t2_data, colLabels=t2_cols, cellLoc="center",
                     loc="center",
                     colWidths=[0.13, 0.10, 0.13, 0.40, 0.24])
    table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.6)

    for j in range(len(t2_cols)):
        c = table[0, j]; c.set_facecolor(BLUE_DARK)
        c.set_text_props(color=TEXT_LIGHT, fontweight="bold")
        c.set_edgecolor(WHITE); c.set_linewidth(0.8)

    for i in range(1, len(t2_data) + 1):
        is_ret = t2_data[i-1][3].startswith("Yes")
        is_ft  = t2_data[i-1][0].startswith("FT-")
        if is_ft:
            bg = C_VBLUE          # pale blue — fine-tuned in-house
        elif is_ret:
            bg = C_PVBLUE         # very pale blue — retrieval-trained
        else:
            bg = BLUE_LIGHT if i % 2 else BLUE_ALT
        for j in range(len(t2_cols)):
            c = table[i, j]; c.set_facecolor(bg)
            c.set_edgecolor(WHITE); c.set_linewidth(0.5)
            if j == 0:
                fg = (C_BLUE2 if is_ft else
                      C_BLUE1 if is_ret else BLUE_DARK)
                c.set_text_props(fontweight="bold", color=fg)
            elif j == 4 and "Limited" in t2_data[i-1][4]:
                c.set_text_props(color=WARN_FG, fontweight="bold")
            elif j == 3 and is_ret and not is_ft:
                c.set_text_props(color=C_BLUE1)
            elif j == 3 and is_ft:
                c.set_text_props(color=C_BLUE2, fontweight="bold")
            else:
                c.set_text_props(color=TEXT_DARK)

    ax.set_title(f"Table 2   Model Characteristics{title_suffix}",
                 fontsize=8, fontweight="bold", color=C_NAVY,
                 pad=6, loc="left")
    caption_extra = " Lighter blue: fine-tuned in this work." if any(
        t2_data[i-1][0].startswith("FT-") for i in range(1, len(t2_data)+1)) else ""
    ax.text(0, 0.985,
            "Pale blue: retrieval-trained off-the-shelf (contrastive / "
            "dual-encoder / translation-pair). "
            "Red 'Limited': BM25 lexical matching degrades for "
            f"non-whitespace scripts.{caption_extra}",
            transform=ax.transAxes, fontsize=5.5, color=C_MGREY,
            style="italic")
    plt.tight_layout()
    _save(fig, outdir, basename)
    plt.show()

    # LaTeX export
    t2_lines = [r"\begin{table}[t]", r"\centering",
                r"\caption{Model characteristics for the retrieval baselines "
                r"evaluated on GeoQueryCA. "
                r"Retrieval-trained = explicitly trained with retrieval or "
                r"sentence-similarity objectives.}",
                r"\label{tab:models}", r"\small",
                r"\begin{tabular}{lccll}", r"\toprule",
                " & ".join(t2_cols) + r" \\", r"\midrule"]
    for row in t2_data:
        t2_lines.append(" & ".join(_tex_escape(c) for c in row) + r" \\")
    t2_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_name = basename.replace("table2_model_characteristics", "table2") \
                       .replace("_with_finetuned", "_full")
    with open(os.path.join(outdir, tex_name + ".tex"), "w") as f:
        f.write("\n".join(t2_lines))

def _save(fig, outdir, name):
    """Save figure and auto-download in Colab, or write to disk locally.

    DPI behaviour
    ─────────────
    • Publication export (outdir files) : 600 dpi  (SN_DPI — journal quality)
    • Auto-download PNG for the user    : 300 dpi  (lighter file, screen-ready)

    Formats written to outdir
    ─────────────────────────
    • .pdf   — vector (preferred for submission)
    • .png   — 600 dpi raster
    • .tiff  — 600 dpi LZW compressed (required by some Nature Portfolio journals)

    Colab auto-download
    ───────────────────
    After saving, triggers google.colab.files.download() on the 300 dpi PNG so
    it lands in the user's Downloads folder automatically.  Silently skipped
    when running outside Colab.
    """
    DOWNLOAD_DPI = 300

    os.makedirs(outdir, exist_ok=True)

    # ── 1. Save all publication formats at SN_DPI ─────────────────────────
    saved_paths = {}
    for ext, kw in [
        ("pdf",  {"backend": "pdf"}),
        ("png",  {"dpi": SN_DPI}),
        ("tiff", {"dpi": SN_DPI, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        path = os.path.join(outdir, f"{name}.{ext}")
        try:
            fig.savefig(path, bbox_inches="tight", facecolor=C_WHITE, **kw)
        except Exception:
            fig.savefig(path, bbox_inches="tight",
                        facecolor=C_WHITE, dpi=SN_DPI)
        saved_paths[ext] = path

    # ── 2. Write a separate 300 dpi PNG for download ──────────────────────
    dl_path = os.path.join(outdir, f"{name}_300dpi.png")
    fig.savefig(dl_path, dpi=DOWNLOAD_DPI,
                bbox_inches="tight", facecolor=C_WHITE)
    saved_paths["300dpi_png"] = dl_path

    print(f"  saved: {outdir}/{name}  "
          f"[pdf | png @{SN_DPI} dpi | tiff @{SN_DPI} dpi | "
          f"300dpi-png for download]")

    # ── 3. Auto-download the 300 dpi PNG in Colab ─────────────────────────
    try:
        from google.colab import files as _colab_files
        _colab_files.download(dl_path)
        print(f"  ↓  auto-download triggered: {os.path.basename(dl_path)}")
    except ImportError:
        pass   # not in Colab — files are on disk, nothing to trigger
    except Exception as _e:
        print(f"  ⚠  auto-download skipped ({_e})")

def _jsonsafe(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    if isinstance(o, dict):           return {str(k): _jsonsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):  return [_jsonsafe(x) for x in o]
    if isinstance(o, set):            return list(o)
    return o

# ════════════════════════════════════════════════════════════════════════════
# Shared geo-error metric functions  (used by Phase 1 profiling AND
# Phase 2 retrieval benchmark — defined here so both phases can call them)
# ════════════════════════════════════════════════════════════════════════════
def geo_error_metrics_from_ranks(ranks, lat_arr, lon_arr):
    """Given a per-query rank array and coordinate arrays (ground-truth order),
    compute coordinate-aware error metrics.

    For each query i the retrieved document is the one at position ranks[i]-1
    after sorting by descending score. Since the score matrix diagonal is the
    ground-truth, rank=1 means the correct document was top-1 (distance = 0 km).

    Metrics returned:
      MAE_geo_km      — Mean geodetic distance between retrieved and true coords
      MedAE_geo_km    — Median geodetic distance
      RMSE_geo_km     — Root-mean-square geodetic distance
      GeoAcc@1km      — Fraction of queries with top-1 retrieved coord within 1 km
      GeoAcc@10km     — Fraction within 10 km
      GeoAcc@50km     — Fraction within 50 km
      GeoAcc@161km    — Fraction within 161 km  (~100 miles, common geo-IR threshold)
      weighted_MRR_geo— MRR weighted by inverse geodetic error (rank-1 perfect=1)
      p90_error_km    — 90th-percentile error (tail behaviour)
    """
    N_q = len(ranks)
    valid = np.isfinite(lat_arr) & np.isfinite(lon_arr)

    # Ground-truth coords for every query (diagonal of the N×N matrix)
    gt_lat = lat_arr.copy()
    gt_lon = lon_arr.copy()

    # Coords of the top-1 retrieved doc (index = rank-1 in the sorted order).
    # rank array gives the 1-based rank of the ground-truth doc, NOT the index
    # of the retrieved doc. We need the index of the HIGHEST-ranked doc.
    # Since we don't have the full score matrix here, we approximate:
    #   * rank == 1 → top-1 IS the ground truth → distance 0
    #   * rank  > 1 → top-1 is some other doc; use mean displacement of wrong docs
    #                 (conservative proxy when the full matrix is unavailable)
    # When called from Phase 2 with the score matrix available (see below), the
    # exact top-1 index is used instead.
    dist_km = np.where(ranks == 1, 0.0, np.nan)   # filled below if exact coords known

    out = {}
    valid_dist = dist_km[~np.isnan(dist_km)]
    rank1_count = int((ranks == 1).sum())

    # Basic geo accuracy at multiple thresholds (rank-1 = distance 0)
    for thresh in (1, 10, 50, 161):
        out[f"GeoAcc@{thresh}km"] = float(rank1_count / N_q)  # lower bound

    out["rank1_exact_frac"] = float(rank1_count / N_q)

    # For distance-based metrics we need exact retrieval coords — see
    # geo_error_metrics_from_score_matrix() for the full version.
    out["_note"] = ("Approximate: distance metrics require score matrix. "
                    "Use geo_error_metrics_from_score_matrix() for full metrics.")
    return out


def geo_error_metrics_from_score_matrix(S_np, lat_arr, lon_arr, ranks,
                                         thresholds=(1, 10, 50, 161)):
    """Full coordinate-aware error metrics computed directly from the N×N score
    matrix S_np (numpy float32/64) so the exact top-1 retrieved index is known.

    Parameters
    ----------
    S_np      : np.ndarray, shape (N, N), score[i, j] = similarity(query_i, passage_j)
    lat_arr   : np.ndarray, shape (N,)   ground-truth latitudes  (decimal degrees)
    lon_arr   : np.ndarray, shape (N,)   ground-truth longitudes (decimal degrees)
    ranks     : np.ndarray, shape (N,)   1-based rank of true doc for each query
    thresholds: tuple of km thresholds for GeoAcc@k

    Returns dict with all geo-error metrics.
    """
    N_q = len(ranks)
    # Index of the top-1 retrieved document for each query
    top1_idx = np.argmax(S_np, axis=1)           # shape (N,)

    gt_lat  = lat_arr                             # ground-truth lat of query's true doc
    gt_lon  = lon_arr
    ret_lat = lat_arr[top1_idx]                   # lat of the top-1 retrieved doc
    ret_lon = lon_arr[top1_idx]

    # Only compute where both gt and retrieved coords are valid
    valid = (np.isfinite(gt_lat) & np.isfinite(gt_lon)
             & np.isfinite(ret_lat) & np.isfinite(ret_lon))
    N_valid = int(valid.sum())

    if N_valid == 0:
        return {"_note": "No valid coordinate pairs — all geo metrics are NaN.",
                "N_valid": 0}

    dist = haversine_km(gt_lat[valid], gt_lon[valid],
                        ret_lat[valid], ret_lon[valid])

    out = {
        "N_valid":          N_valid,
        "MAE_geo_km":       float(np.mean(dist)),
        "MedAE_geo_km":     float(np.median(dist)),
        "RMSE_geo_km":      float(np.sqrt(np.mean(dist**2))),
        "p90_error_km":     float(np.percentile(dist, 90)),
        "p95_error_km":     float(np.percentile(dist, 95)),
        "max_error_km":     float(np.max(dist)),
        "min_error_km":     float(np.min(dist)),
    }
    for t in thresholds:
        out[f"GeoAcc@{t}km"] = float(np.mean(dist <= t))

    # Geo-weighted MRR: queries with rank=1 AND correct coords get full credit;
    # others are penalised by distance. Specifically:
    #   geo_weight(i) = exp(-dist_km(i) / 100)   (100 km half-life)
    # Then geo_MRR = mean( geo_weight(i) / rank(i) )
    r = ranks.astype(np.float64)
    #geo_w = np.ones(N_q)
    geo_w_valid = np.exp(-dist / 100.0)
    out["geo_weighted_MRR"] = float(np.mean(geo_w_valid / r[valid]))
    out["geo_weighted_MRR_N"] = int(valid.sum())

    # Per-rank-bucket mean error: how far off are rank-2, rank-3, ... retrievals?
    rank_buckets = {}
    for bucket_label, mask in [("rank_1",   ranks == 1),
                                ("rank_2_5", (ranks >= 2) & (ranks <= 5)),
                                ("rank_6+",  ranks >= 6)]:
        bm = mask & valid
        if bm.sum() > 0:
            rank_buckets[bucket_label] = {
                "n":          int(bm.sum()),
                "mean_km":    float(np.mean(haversine_km(
                                  gt_lat[bm], gt_lon[bm],
                                  lat_arr[np.argmax(S_np[bm], axis=1)],
                                  lon_arr[np.argmax(S_np[bm], axis=1)]))),
            }
    out["error_by_rank_bucket"] = rank_buckets
    return out


def geo_error_figure(geo_results_dict, outdir, basename,
                     title="Geo-Error Metrics by Model"):
    """4-panel figure: MAE bar, GeoAcc@k grouped bar, error CDF, geo-weighted MRR bar."""
    labels = [lab for lab, g in geo_results_dict.items()
              if "MAE_geo_km" in g]
    if not labels:
        print("  [geo fig] skipped — no MAE data available")
        return

    mae     = [geo_results_dict[l]["MAE_geo_km"]      for l in labels]
    medae   = [geo_results_dict[l]["MedAE_geo_km"]    for l in labels]
    rmse    = [geo_results_dict[l]["RMSE_geo_km"]      for l in labels]
    gwmrr   = [geo_results_dict[l].get("geo_weighted_MRR", float("nan")) for l in labels]
    acc1    = [geo_results_dict[l].get("GeoAcc@1km",   0) for l in labels]
    acc10   = [geo_results_dict[l].get("GeoAcc@10km",  0) for l in labels]
    acc50   = [geo_results_dict[l].get("GeoAcc@50km",  0) for l in labels]
    acc161  = [geo_results_dict[l].get("GeoAcc@161km", 0) for l in labels]

    fig, axes = plt.subplots(2, 2, figsize=(SN_2COL, SN_2COL * 0.72), dpi=SN_DPI)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=BLUE_DARK, y=1.01)

    # (a) MAE / MedAE / RMSE bar cluster
    ax = axes[0, 0]
    x = np.arange(len(labels)); w = 0.25
    ax.bar(x - w, mae,   width=w, color=PALETTE[0], label="MAE",   edgecolor="white")
    ax.bar(x,     medae, width=w, color=PALETTE[1], label="MedAE", edgecolor="white")
    ax.bar(x + w, rmse,  width=w, color=PALETTE[2], label="RMSE",  edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Distance error (km)"); ax.set_title("(a) Geodetic Error — MAE / MedAE / RMSE")
    ax.legend(fontsize=8); ax.yaxis.grid(True, color="#E5E5E5", lw=0.5); ax.set_axisbelow(True)

    # (b) GeoAcc@k grouped bar
    ax = axes[0, 1]
    x = np.arange(len(labels)); w = 0.18
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    for off, accs, label_k, col in zip(offsets,
                                        [acc1, acc10, acc50, acc161],
                                        ["@1km","@10km","@50km","@161km"],
                                        PALETTE[:4]):
        bars = ax.bar(x + off, [a*100 for a in accs], width=w,
                      color=col, label=f"GeoAcc{label_k}", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("(b) GeoAcc at Distance Thresholds")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper left"); ax.yaxis.grid(True, color="#E5E5E5", lw=0.5)
    ax.set_axisbelow(True)

    # (c) p90 / p95 / max error bar
    ax = axes[1, 0]
    p90 = [geo_results_dict[l].get("p90_error_km", float("nan")) for l in labels]
    p95 = [geo_results_dict[l].get("p95_error_km", float("nan")) for l in labels]
    mx  = [geo_results_dict[l].get("max_error_km", float("nan")) for l in labels]
    x = np.arange(len(labels)); w = 0.25
    ax.bar(x - w, p90, width=w, color=PALETTE[3], label="p90",  edgecolor="white")
    ax.bar(x,     p95, width=w, color=PALETTE[4], label="p95",  edgecolor="white")
    ax.bar(x + w, mx,  width=w, color=PALETTE[5], label="max",  edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Distance error (km)"); ax.set_title("(c) Tail Error — p90 / p95 / Max")
    ax.legend(fontsize=8); ax.yaxis.grid(True, color="#E5E5E5", lw=0.5); ax.set_axisbelow(True)

    # (d) Geo-weighted MRR
    ax = axes[1, 1]
    valid_idx = [i for i, v in enumerate(gwmrr) if not math.isnan(v)]
    if valid_idx:
        v_labels = [labels[i] for i in valid_idx]
        v_gwmrr  = [gwmrr[i]  for i in valid_idx]
        bars = ax.bar(range(len(v_labels)), v_gwmrr,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(v_labels))],
                      edgecolor="white", width=0.6)
        for b, v in zip(bars, v_gwmrr):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                    f"{v:.4f}", ha="center", fontsize=6, color=C_NAVY,
                    fontweight="bold")
        ax.set_xticks(range(len(v_labels)))
        ax.set_xticklabels(v_labels, fontsize=9, fontweight="bold")
        ax.set_ylabel("Geo-weighted MRR")
        ax.set_title("(d) Geo-weighted MRR  (distance-penalised)")
        ax.set_ylim(0, max(v_gwmrr) * 1.18)
        ax.yaxis.grid(True, color="#E5E5E5", lw=0.5); ax.set_axisbelow(True)
        ax.text(0.02, 0.97,
                "Weight = exp(−dist/100 km); rank-1 correct retrieval = 1.0",
                transform=ax.transAxes, fontsize=7.5, color="#666666",
                style="italic", va="top")
    else:
        ax.set_visible(False)

    plt.tight_layout()
    _save(fig, outdir, basename)
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#                       PHASE 1 — DATASET PROFILING
# ════════════════════════════════════════════════════════════════════════════
profiling_summary = None
benchmark_summary = None
finetune_summary  = None
if CONFIG["run_profiling"]:
    print("\n" + "═"*72 + "\nPHASE 1 — Dataset Profiling\n" + "═"*72)

    # ── 3.1 Compute metrics ──────────────────────────────────────────────
    def shannon_entropy(s):
        p = s.value_counts(normalize=True).values
        return float(entropy(p, base=2))
    def normalised_entropy(s):
        k = s.nunique()
        return float(shannon_entropy(s) / math.log2(k)) if k > 1 else 0.0
    def gini(s):
        counts = np.sort(s.value_counts().values)
        n = len(counts)
        if n == 0: return 0.0
        cum = np.cumsum(counts, dtype=float)
        return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

    metrics = {"dataset": {"source_file": os.path.basename(CSV_PATH),
                           "n_records": int(N),
                           "n_columns": int(df.shape[1])}}
    quality = {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_monuments": int(df["Monument"].duplicated().sum())
                               if "Monument" in df else None,
        "completeness_overall_pct":
            float(100 * (1 - df.isna().sum().sum() / df.size)),
        "missing_per_column":
            {c: int(df[c].isna().sum()) for c in df.columns},
        "completeness_per_column_pct":
            {c: float(100 * (1 - df[c].isna().mean())) for c in df.columns},
    }
    metrics["quality"] = quality

    # ── Coordinate coverage ──────────────────────────────────────────────
    if HAS_COORDS:
        lat_ok = int(df["Lat_DD"].notna().sum())
        lon_ok = int(df["Lon_DD"].notna().sum())
        metrics["coordinates"] = {
            "has_coordinates":        True,
            "lat_dd_valid_n":         lat_ok,
            "lon_dd_valid_n":         lon_ok,
            "lat_dd_valid_pct":       float(100 * lat_ok / N),
            "lon_dd_valid_pct":       float(100 * lon_ok / N),
            "lat_range":              [float(df["Lat_DD"].min()), float(df["Lat_DD"].max())],
            "lon_range":              [float(df["Lon_DD"].min()), float(df["Lon_DD"].max())],
            "lat_mean":               float(df["Lat_DD"].mean()),
            "lon_mean":               float(df["Lon_DD"].mean()),
            "lat_std":                float(df["Lat_DD"].std()),
            "lon_std":                float(df["Lon_DD"].std()),
        }
        print(f"Coordinate coverage  ..  Lat {lat_ok}/{N}  Lon {lon_ok}/{N}")
    else:
        metrics["coordinates"] = {"has_coordinates": False}

    diversity = {}
    for col in ["Country","Region","Query_Language","Category","City"]:
        if col in df.columns:
            diversity[col] = {
                "unique": int(df[col].nunique()),
                "entropy_bits": shannon_entropy(df[col]),
                "normalised_entropy": normalised_entropy(df[col]),
                "gini": gini(df[col]),
                "top1_share_pct": float(100 * df[col].value_counts(normalize=True).iloc[0]),
            }
    metrics["diversity"] = diversity

    metrics["geographic"] = {
        "by_region": df["Region"].value_counts().to_dict(),
        "by_region_pct": (df["Region"].value_counts(normalize=True) * 100)
                           .round(2).to_dict(),
        "top_countries": df["Country"].value_counts().head(15).to_dict(),
    }
    metrics["language"] = {
        "by_language": df["Query_Language"].value_counts().to_dict(),
        "top_languages": df["Query_Language"].value_counts().head(15).to_dict(),
    }
    metrics["category"] = {
        "by_category": df["Category"].value_counts().to_dict(),
        "top_categories": df["Category"].value_counts().head(10).to_dict(),
    }

    if "Confidence_Score" in df.columns:
        cs = df["Confidence_Score"].dropna()
        metrics["confidence_score"] = {
            "n": int(cs.size),
            "mean": float(cs.mean()), "std": float(cs.std()),
            "min": float(cs.min()),   "max": float(cs.max()),
            "median": float(cs.median()),
            "q1": float(cs.quantile(0.25)), "q3": float(cs.quantile(0.75)),
            "pct_ge_0_95": float(100 * (cs >= 0.95).mean()),
            "pct_ge_0_97": float(100 * (cs >= 0.97).mean()),
        }

    metrics["crosstabs"] = {
        "region_x_category":
            pd.crosstab(df["Region"], df["Category"]).to_dict(),
        "region_x_language_top10":
            pd.crosstab(df["Region"],
                df["Query_Language"].where(
                    df["Query_Language"].isin(
                        df["Query_Language"].value_counts().head(10).index),
                    other="Other")).to_dict(),
    }

    print(f"Records ............... {N:,}")
    print(f"Overall completeness .. {quality['completeness_overall_pct']:.2f}%")
    print(f"Unique countries ...... {diversity['Country']['unique']}")
    print(f"Unique languages ...... {diversity['Query_Language']['unique']}")
    print(f"Unique categories ..... {diversity['Category']['unique']}")
    print(f"Region entropy (norm).. {diversity['Region']['normalised_entropy']:.3f}")
    if "confidence_score" in metrics:
        c = metrics["confidence_score"]
        print(f"Confidence μ ± σ ...... {c['mean']:.3f} ± {c['std']:.3f}")
        print(f"  ≥ 0.95 .............. {c['pct_ge_0_95']:.1f}%")

    # ── 3.2 Figures (6) ──────────────────────────────────────────────────
    # ── Shared pie helper ─────────────────────────────────────────────────
    def _draw_publication_pie(ax, vals_series, colors, title,
                              caption, explode_first=True,
                              pct_inside_threshold=12.0):
        """Draw a publication-quality pie chart matching the reference style:
        • Large wedges  (≥ threshold %): bold percentage INSIDE the wedge
        • Small wedges  (< threshold %): percentage outside with a fine leader line
        • Legend        : below the chart, 2-column, label + count + %
        • No chart border; white wedge edges; tight bold title
        """
        total   = vals_series.sum()
        pcts    = vals_series / total * 100
        explode = ([0.04] + [0.0] * (len(vals_series) - 1)) if explode_first else \
                  [0.0] * len(vals_series)

        wedges, _ = ax.pie(
            vals_series,
            labels=None, autopct=None,
            startangle=90,
            colors=colors,
            explode=explode,
            wedgeprops=dict(linewidth=0.8, edgecolor=C_WHITE),
            counterclock=False,
            radius=1.0,
        )

        # ── Percentage labels ─────────────────────────────────────────────
        for wedge, (lbl, val) in zip(wedges, vals_series.items()):
            pct   = pcts[lbl]
            ang   = (wedge.theta1 + wedge.theta2) / 2.0
            ang_r = math.radians(ang)
            r_mid = 0.65   # distance from centre for inside label

            if pct >= pct_inside_threshold:
                # Inside the wedge — bold white percentage
                ax.text(r_mid * math.cos(ang_r),
                        r_mid * math.sin(ang_r),
                        f"{pct:.1f}%",
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color=C_WHITE)
            else:
                # Outside with a thin grey leader line
                r_out = 1.18
                r_txt = 1.32
                ax.annotate(
                    f"{pct:.1f}%",
                    xy=(r_out * math.cos(ang_r), r_out * math.sin(ang_r)),
                    xytext=(r_txt * math.cos(ang_r), r_txt * math.sin(ang_r)),
                    ha="left" if math.cos(ang_r) >= 0 else "right",
                    va="center",
                    fontsize=6, color=C_DGREY,
                    arrowprops=dict(arrowstyle="-", color=C_MGREY,
                                   lw=0.5),
                )

        # ── Legend BELOW the chart ────────────────────────────────────────
        legend_labels = [
            f"{lbl}  {int(v):,} ({pcts[lbl]:.1f}%)"
            for lbl, v in vals_series.items()
        ]
        ax.legend(
            handles=[mpatches.Patch(facecolor=c, edgecolor=C_WHITE,
                                    linewidth=0.5)
                     for c in colors],
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=2,
            fontsize=5.5,
            frameon=False,
            handlelength=0.9,
            handleheight=0.8,
            borderpad=0.3,
            labelspacing=0.25,
            columnspacing=0.8,
        )

        ax.set_title(title, fontsize=7, fontweight="bold",
                     color=C_NAVY, pad=8, wrap=True)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        # Caption below legend
        ax.text(0, -1.52, caption, transform=ax.transData,
                ha="center", va="top", fontsize=5.5,
                fontstyle="italic", color=C_DGREY)

    # ── Figure 1a — Geographic Distribution pie ───────────────────────────
    print("\n[1/8] Geographic distribution pie")
    ind_mask = df["Country"].str.lower() == "india"
    ind_n    = int(ind_mask.sum())
    rest     = df.loc[~ind_mask, "Region"].value_counts()
    geo_vals = pd.concat([pd.Series({"India (over-sampled)": ind_n}), rest])
    geo_pcts = geo_vals / geo_vals.sum() * 100

    # Colours matching Image 1: dark navy → mid-blue → steel-blue → pale blues
    GEO_COLORS = ["#1F3864",   # India        — darkest navy
                  "#2E5F9E",   # Asia excl.   — dark blue
                  "#4472C4",   # Europe       — mid blue
                  "#7BAFD4",   # Americas     — steel blue
                  "#9DC3E6",   # Africa       — pale blue
                  "#BDD7EE",   # Global/Other — very pale blue
                  "#DEEAF1"]   # extra        — lightest
    geo_colors = [GEO_COLORS[i % len(GEO_COLORS)]
                  for i in range(len(geo_vals))]

    fig1a, ax1a = plt.subplots(
        figsize=(SN_15COL, SN_15COL * 1.25), dpi=SN_DPI)
    fig1a.patch.set_facecolor(C_WHITE)

    _draw_publication_pie(
        ax1a, geo_vals, geo_colors,
        title=f"Geographic Distribution of GeoQueryCA Records\nby Macro-Region (N = {N:,})",
        caption=f"Fig. 1: Geographic Distribution of GeoQueryCA Records by Macro-Region",
        pct_inside_threshold=10.0,
    )
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    _save(fig1a, DESC_DIR, "fig1a_geographic_pie")
    plt.show()

    # ── Figure 1b — Resource Tier Distribution pie ────────────────────────
    print("[2/8] Resource tier pie")
    # Classify languages into resource tiers
    # High-resource  : English + major European langs (>10M speakers, good NLP)
    # Mid-resource   : Indian vernaculars + other major langs
    # Low-resource   : remaining / under-resourced langs
    HIGH_RESOURCE_LANGS = {
        "English","Spanish","French","German","Italian","Portuguese",
        "Dutch","Polish","Russian","Chinese","Japanese","Korean","Arabic",
        "Turkish","Romanian","Czech","Greek","Swedish","Hungarian",
    }
    MID_RESOURCE_LANGS = INDIAN_VERNACULARS | {
        "Indonesian","Vietnamese","Thai","Persian","Ukrainian","Catalan",
        "Danish","Norwegian","Finnish","Bulgarian","Croatian","Slovak",
    }

    def _resource_tier(lang):
        if lang in HIGH_RESOURCE_LANGS: return "High-Resource"
        if lang in MID_RESOURCE_LANGS:  return "Mid-Resource"
        return "Low-Resource"

    df["Resource_Tier"] = df["Query_Language"].apply(_resource_tier)
    tier_counts = df["Resource_Tier"].value_counts().reindex(
        ["High-Resource", "Mid-Resource", "Low-Resource"]).fillna(0).astype(int)

    # Colours matching Image 2: navy, mid-blue, pale blue
    TIER_COLORS = ["#1F3864",   # High-Resource — dark navy
                   "#4472C4",   # Mid-Resource  — mid blue
                   "#BDD7EE"]   # Low-Resource  — pale blue

    fig1b, ax1b = plt.subplots(
        figsize=(SN_15COL, SN_15COL * 1.25), dpi=SN_DPI)
    fig1b.patch.set_facecolor(C_WHITE)

    _draw_publication_pie(
        ax1b, tier_counts, TIER_COLORS,
        title=f"Resource Tier Distribution of GeoQueryCA (N={N:,})",
        caption=f"Fig. 3: Resource Tier Distribution of GeoQueryCA (N={N:,})",
        pct_inside_threshold=10.0,
    )
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    _save(fig1b, DESC_DIR, "fig1b_resource_tier_pie")
    plt.show()

    # Figure 2 — 4-panel statistical profile
    print("[3/8] 4-panel statistical profile")

    # ── Colour scheme exactly matching the reference image ────────────────
    # (a) Category bar:  dark-navy tallest bars, stepping through blue shades
    # (b) Language bar:  4 groups — English (dark navy), Indian vernaculars
    #                    (mid-blue), European (light blue), Others (grey-blue)
    # (c) Confidence:    two-tone — dark bars for main mass, lighter for tails
    #                    red italic μ annotation
    #
    # Layout: (a) full-width top row, (b) bottom-left, (c) bottom-right
    # Font:   value labels ABOVE/BESIDE each bar, bold for large values

    # ── Colour definitions matching the figure ────────────────────────────
    _COL_ENG  = "#1F3864"   # English        — darkest navy
    _COL_IND  = "#2E5F9E"   # Indian vernac  — mid-dark blue
    _COL_EUR  = "#9DC3E6"   # European       — pale blue
    _COL_OTH  = "#BDD7EE"   # Others         — very pale blue / grey-blue
    _COL_BAR  = "#2E5F9E"   # default bar fill (category chart)
    _COL_DARK = "#1F3864"   # tallest bars (accent)
    _COL_HIST_DARK = "#2E5F9E"
    _COL_HIST_LITE = "#9DC3E6"
    _COL_MU   = "#C00000"   # red μ line + label

    # Assign language-group colour
    def _lang_color(name):
        if name == "English":             return _COL_ENG
        if name in INDIAN_VERNACULARS:    return _COL_IND
        if name in EUROPEAN_LANGS:        return _COL_EUR
        return _COL_OTH

    # ── Figure layout: 3 panels with GridSpec ─────────────────────────────
    # top row: (a) full width
    # bottom row: (b) left 55%, (c) right 45%
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(SN_2COL, SN_2COL * 0.85), dpi=SN_DPI)
    fig.patch.set_facecolor(C_WHITE)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[1.0, 1.1],
                           width_ratios=[1.1, 1.0],
                           hspace=0.52, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])   # (a) spans both columns
    ax_b = fig.add_subplot(gs[1, 0])   # (b) bottom-left
    ax_c = fig.add_subplot(gs[1, 1])   # (c) bottom-right

    # ── (a) Monument Category Distribution ───────────────────────────────
    cat_n = min(10, df["Category"].nunique())
    cat   = df["Category"].value_counts().head(cat_n)
    # Colour: top-2 bars in darkest navy, rest in mid-blue (matches image)
    cat_colors = [_COL_DARK if i < 2 else _COL_BAR for i in range(len(cat))]
    x_pos = range(len(cat))
    bars = ax_a.bar(x_pos, cat.values,
                    color=cat_colors, edgecolor=C_WHITE,
                    width=0.65, linewidth=0.3)
    # Value labels above each bar
    for b, v in zip(bars, cat.values):
        ax_a.text(b.get_x() + b.get_width() / 2,
                  b.get_height() + cat.values.max() * 0.012,
                  f"{v:,}", ha="center", va="bottom",
                  fontsize=6, fontweight="bold", color=C_DGREY)
    ax_a.set_xticks(list(x_pos))
    ax_a.set_xticklabels(cat.index, rotation=30, ha="right", fontsize=6)
    ax_a.set_ylabel("Number of Records", fontsize=6)
    ax_a.set_ylim(0, cat.values.max() * 1.20)
    ax_a.set_title("(a) Monument Category Distribution",
                   fontsize=7, fontweight="bold", loc="center", pad=4)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.yaxis.grid(True, color=C_LGREY, linewidth=0.35, alpha=0.7)
    ax_a.set_axisbelow(True)

    # ── (b) Query Language Distribution ──────────────────────────────────
    lang_n  = min(12, df["Query_Language"].nunique())
    lang_s  = df["Query_Language"].value_counts().head(lang_n).iloc[::-1]
    # Group label appended for non-English/Indian  (matches "(9)" style in image)
    grp_counts = {"English": 0, "Indian vernaculars": 0,
                  "European": 0, "Others": 0}
    for nm in lang_s.index:
        grp_counts[lang_group(nm)] += 1

    def _lang_ylabel(name):
        g = lang_group(name)
        if g == "European":
            return f"European ({grp_counts['European']})"
        if g == "Others":
            return f"Others ({grp_counts['Others']})"
        return name

    yticklabels = [_lang_ylabel(nm) for nm in lang_s.index]
    bar_colors  = [_lang_color(nm) for nm in lang_s.index]

    bars = ax_b.barh(range(len(lang_s)), lang_s.values,
                     color=bar_colors, edgecolor=C_WHITE,
                     height=0.60, linewidth=0.3)
    ax_b.set_yticks(range(len(lang_s)))
    ax_b.set_yticklabels(yticklabels, fontsize=5.5)
    for b, v in zip(bars, lang_s.values):
        ax_b.text(b.get_width() + lang_s.values.max() * 0.02,
                  b.get_y() + b.get_height() / 2,
                  f"{v:,}", va="center", fontsize=6,
                  fontweight="bold", color=C_DGREY)
    ax_b.set_xlim(0, lang_s.values.max() * 1.28)
    ax_b.set_xlabel("Number of Records", fontsize=6)
    ax_b.set_title("(b) Query Language Distribution",
                   fontsize=7, fontweight="bold", loc="center", pad=4)
    ax_b.spines["left"].set_visible(False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    ax_b.xaxis.grid(True, color=C_LGREY, linewidth=0.35, alpha=0.7)
    ax_b.set_axisbelow(True)
    # 4-colour legend matching image
    ax_b.legend(
        handles=[
            mpatches.Patch(color=_COL_ENG,  label="English"),
            mpatches.Patch(color=_COL_EUR,  label="European"),
            mpatches.Patch(color=_COL_IND,  label="Indian vernaculars"),
            mpatches.Patch(color=_COL_OTH,  label="Others"),
        ],
        fontsize=5, ncol=2, loc="lower right",
        frameon=True, edgecolor=C_LGREY, framealpha=0.9,
        handlelength=0.8, borderpad=0.4, labelspacing=0.3,
    )

    # ── (c) Confidence Score Distribution ────────────────────────────────
    if "Confidence_Score" in df.columns and df["Confidence_Score"].notna().any():
        cs = df["Confidence_Score"].dropna().values
        # Fixed bins matching the image tick labels (0.93–0.99)
        _bins = np.arange(cs.min() - 0.005, cs.max() + 0.015, 0.01)
        n_arr, bin_edges, patches_c = ax_c.hist(
            cs, bins=_bins,
            edgecolor=C_WHITE, linewidth=0.3, rwidth=0.80)
        # Two-tone: bars at/above μ are darker (matches image)
        mu = float(np.mean(cs))
        for p, left in zip(patches_c, bin_edges[:-1]):
            p.set_facecolor(_COL_HIST_DARK if left >= mu - 0.005
                            else _COL_HIST_LITE)
        # Value labels on top of each bar
        for p, v in zip(patches_c, n_arr):
            if v > 0:
                ax_c.text(p.get_x() + p.get_width() / 2,
                          v + n_arr.max() * 0.012,
                          f"{int(v):,}", ha="center", va="bottom",
                          fontsize=5.5, fontweight="bold", color=C_DGREY)
        # μ vertical line — red dashed, italic label top-right
        ax_c.axvline(mu, color=_COL_MU, linestyle="--", linewidth=0.9)
        ax_c.text(mu + (cs.max() - cs.min()) * 0.04,
                  n_arr.max() * 0.88,
                  f"μ = {mu:.2f}",
                  color=_COL_MU, fontsize=6.5,
                  fontstyle="italic", fontweight="bold")
        ax_c.set_xlabel("Confidence Score", fontsize=6)
        ax_c.set_ylabel("Frequency", fontsize=6)
        ax_c.set_title("(c) Confidence Score Distribution",
                       fontsize=7, fontweight="bold", loc="center", pad=4)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)
        ax_c.yaxis.grid(True, color=C_LGREY, linewidth=0.35, alpha=0.7)
        ax_c.set_axisbelow(True)
        # Match x-tick label style from image
        ax_c.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax_c.tick_params(axis="x", labelsize=5.5, rotation=45)
    else:
        ax_c.set_visible(False)

    # ── Figure caption (matches "Fig. 2: GeoQueryCA Statistical Profiling") ──
    fig.text(0.5, -0.01,
             f"Fig. 2: GeoQueryCA Statistical Profiling  (N = {N:,})",
             ha="center", fontsize=6.5, fontstyle="italic", color=C_DGREY)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, DESC_DIR, "fig2_statistical_profile")
    plt.show()

    # Figure 3 — Top countries
    print("[4/8] Top countries")
    top_n = min(15, df["Country"].nunique())
    top_c = df["Country"].value_counts().head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(SN_15COL, SN_15COL * 0.80), dpi=SN_DPI)
    bars = ax.barh(top_c.index, top_c.values,
                   color=[C_NAVY if name == "India" else C_BLUE3
                          for name in top_c.index],
                   edgecolor="white", height=0.65)
    for b, v in zip(bars, top_c.values):
        ax.text(b.get_width() + top_c.values.max() * 0.01,
                b.get_y() + b.get_height() / 2,
                f"{v:,} ({v / N * 100:.1f}%)", va="center", fontsize=6)
    ax.set_xlim(0, top_c.values.max() * 1.18)
    ax.set_xlabel("Number of Records")
    ax.set_title(f"Top {top_n} Countries by Record Count (N = {N:,})", pad=6)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    plt.tight_layout(); _save(fig, DESC_DIR, "fig3_top_countries"); plt.show()

    # Figure 4 — Region × Category heatmap
    print("[5/8] Region × Category heatmap")
    top_cats = df["Category"].value_counts().head(10).index
    sub = df[df["Category"].isin(top_cats)]
    ct = pd.crosstab(sub["Region"], sub["Category"])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index,
               ct.sum(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(SN_2COL, SN_2COL * 0.42), dpi=SN_DPI)
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues",
                cbar_kws={"label":"Records", "shrink": 0.8},
                linewidths=0.3, linecolor="white", ax=ax,
                annot_kws={"fontsize": 6})
    ax.set_title(f"Region × Category Cross-Tabulation (top {len(top_cats)} categories)",
                 pad=6)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    plt.tight_layout(); _save(fig, DESC_DIR, "fig4_region_category_heatmap"); plt.show()

    # Figure 5 — Confidence by region (box+strip)
    if "Confidence_Score" in df.columns and df["Confidence_Score"].notna().any():
        print("[6/8] Confidence by region")
        order = df.groupby("Region")["Confidence_Score"].median()\
                 .sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(SN_15COL, SN_15COL * 0.60), dpi=SN_DPI)
        sns.boxplot(data=df, x="Region", y="Confidence_Score", order=order,
                    ax=ax, palette=PALETTE[:len(order)], width=0.50,
                    fliersize=1.5, linewidth=0.6)
        sns.stripplot(data=df, x="Region", y="Confidence_Score", order=order,
                      ax=ax, color=C_DGREY, alpha=0.15, size=1.2, jitter=0.22)
        medians = df.groupby("Region")["Confidence_Score"].median().reindex(order)
        for i, (_, m) in enumerate(medians.items()):
            ax.text(i, m + 0.003, f"{m:.3f}", ha="center", fontsize=6,
                    color=C_NAVY, fontweight="bold")
        ax.set_title("Confidence Score Distribution by Region", pad=10)
        ax.set_xlabel(""); ax.set_ylabel("Confidence Score")
        ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.5); ax.set_axisbelow(True)
        plt.tight_layout(); _save(fig, DESC_DIR, "fig5_confidence_by_region"); plt.show()
    else:
        print("[6/8] Skipped (no Confidence_Score)")

    # Figure 6 — Diversity metrics
    print("[7/8] Diversity metrics")
    div_cols = [c for c in ["Country","Region","Query_Language","Category","City"]
                if c in df.columns]
    div_df = pd.DataFrame({
        "Field": div_cols,
        "Unique":       [diversity[c]["unique"]              for c in div_cols],
        "Norm.Entropy": [diversity[c]["normalised_entropy"]  for c in div_cols],
        "Top-1 Share %":[diversity[c]["top1_share_pct"]      for c in div_cols],
        "Gini":         [diversity[c]["gini"]                for c in div_cols],
    })
    fig, axes = plt.subplots(1, 2, figsize=(SN_2COL, SN_2COL * 0.36), dpi=SN_DPI)
    ax = axes[0]
    bars = ax.bar(div_df["Field"], div_df["Norm.Entropy"],
                  color=PALETTE[:len(div_df)], edgecolor="white", width=0.6)
    for b, v in zip(bars, div_df["Norm.Entropy"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{v:.2f}", ha="center", fontsize=6)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Normalised entropy (0–1)")
    ax.set_title("Distribution Uniformity by Field")
    ax = axes[1]
    bars = ax.bar(div_df["Field"], div_df["Top-1 Share %"],
                  color=PALETTE[:len(div_df)], edgecolor="white", width=0.6)
    for b, v in zip(bars, div_df["Top-1 Share %"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{v:.1f}%", ha="center", fontsize=6)
    ax.set_ylabel("Share of most common value (%)")
    ax.set_title("Concentration on Most Frequent Value")
    ax.set_ylim(0, max(div_df["Top-1 Share %"]) * 1.18)
    plt.tight_layout(); _save(fig, DESC_DIR, "fig6_diversity_metrics"); plt.show()

    # Figure 7 — Coordinate scatter (geographic distribution of monuments)
    if HAS_COORDS:
        print("[8/8] Coordinate scatter map")
        coord_df = df[df["Lat_DD"].notna() & df["Lon_DD"].notna()].copy()
        fig, axes = plt.subplots(1, 2, figsize=(SN_2COL, SN_2COL * 0.42), dpi=SN_DPI)
        fig.suptitle(f"Spatial Distribution of GeoQueryCA Monuments (N={len(coord_df):,})",
                     fontsize=8)

        # Left: all monuments coloured by Region
        ax = axes[0]
        region_colors = {r: PALETTE[i % len(PALETTE)]
                         for i, r in enumerate(coord_df["Region"].unique())}
        for region, grp in coord_df.groupby("Region"):
            ax.scatter(grp["Lon_DD"], grp["Lat_DD"],
                       c=region_colors[region], s=4, alpha=0.55,
                       label=f"{region} ({len(grp)})", edgecolors="none")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.set_title("(a) All monuments by Region", fontweight="bold")
        ax.legend(fontsize=5.5, markerscale=2, loc="lower left",
                  framealpha=0.85, edgecolor=C_LGREY)
        ax.axhline(0, color=C_MGREY, lw=0.4, ls="--")

        # Right: density hex-bin (spatial concentration)
        ax = axes[1]
        hb = ax.hexbin(coord_df["Lon_DD"], coord_df["Lat_DD"],
                       gridsize=38, cmap="YlOrRd", mincnt=1, linewidths=0.15)
        cb = plt.colorbar(hb, ax=ax, shrink=0.75)
        cb.set_label("Count per hex cell", fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.set_title("(b) Spatial density (hex-bin)", fontweight="bold")
        ax.axhline(0, color=C_MGREY, lw=0.4, ls="--")

        plt.tight_layout(); _save(fig, DESC_DIR, "fig7_coordinate_scatter"); plt.show()
    else:
        print("[8/8] Coordinate scatter skipped (no valid coordinates)")
    with open(os.path.join(DESC_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(_jsonsafe(metrics), f, indent=2, ensure_ascii=False)

    md = [f"# GeoQueryCA — Dataset Profiling Report",
          f"\n**Source:** `{os.path.basename(CSV_PATH)}`  ",
          f"**Records:** {N:,}  ",
          f"**Columns:** {df.shape[1]}\n",
          "## Data quality",
          f"- Overall completeness: **{quality['completeness_overall_pct']:.2f}%**",
          f"- Duplicate rows: **{quality['duplicate_rows']}**"]
    if quality["duplicate_monuments"] is not None:
        md.append(f"- Duplicate monument names: **{quality['duplicate_monuments']}**")
    md += ["\n## Diversity",
           "| Field | Unique | Entropy (bits) | Norm. Entropy | Gini | Top-1 share |",
           "|---|---:|---:|---:|---:|---:|"]
    for c in div_cols:
        d = diversity[c]
        md.append(f"| {c} | {d['unique']} | {d['entropy_bits']:.2f} | "
                  f"{d['normalised_entropy']:.3f} | {d['gini']:.3f} | "
                  f"{d['top1_share_pct']:.1f}% |")
    md += ["\n## Geographic distribution",
           "| Region | Records | Share |", "|---|---:|---:|"]
    for r, v in df["Region"].value_counts().items():
        md.append(f"| {r} | {v:,} | {v / N * 100:.1f}% |")
    if "confidence_score" in metrics:
        c = metrics["confidence_score"]
        md += ["\n## Confidence score",
               f"- Mean ± SD: **{c['mean']:.3f} ± {c['std']:.3f}**",
               f"- Range: [{c['min']:.2f}, {c['max']:.2f}]",
               f"- Records with score ≥ 0.95: **{c['pct_ge_0_95']:.1f}%**",
               f"- Records with score ≥ 0.97: **{c['pct_ge_0_97']:.1f}%**"]
    if HAS_COORDS:
        co = metrics["coordinates"]
        md += ["\n## Coordinate coverage (decimal degrees)",
               f"- Valid Lat_DD: **{co['lat_dd_valid_n']:,} / {N:,}** "
               f"({co['lat_dd_valid_pct']:.1f}%)",
               f"- Valid Lon_DD: **{co['lon_dd_valid_n']:,} / {N:,}** "
               f"({co['lon_dd_valid_pct']:.1f}%)",
               f"- Latitude range:  [{co['lat_range'][0]:.4f}°, {co['lat_range'][1]:.4f}°]",
               f"- Longitude range: [{co['lon_range'][0]:.4f}°, {co['lon_range'][1]:.4f}°]",
               f"- Centroid: ({co['lat_mean']:.4f}°, {co['lon_mean']:.4f}°)"]
    with open(os.path.join(DESC_DIR, "evaluation_report.md"), "w") as f:
        f.write("\n".join(md))
    profiling_summary = {"metrics": metrics, "diversity_cols": div_cols}
    print(f"\nPhase 1 outputs → {DESC_DIR}/")

# ════════════════════════════════════════════════════════════════════════════
#         PHASE 1.5 — COORDINATE GROUND-TRUTH VALIDATION
#         Fetch true coordinates from Wikipedia & Wikidata for each monument,
#         compare against dataset Lat_DD / Lon_DD, compute error statistics,
#         and produce a publication-grade validation figure + CSV.
# ════════════════════════════════════════════════════════════════════════════
coord_validation_summary = None
if CONFIG.get("run_coord_validation") and HAS_COORDS:
    print("\n" + "═"*72 +
          "\nPHASE 1.5 — Coordinate Ground-Truth Validation\n" + "═"*72)

    import requests as _req
    import time as _time

    COORD_VAL_DIR = os.path.join(ROOT, "coord_validation")
    os.makedirs(COORD_VAL_DIR, exist_ok=True)

    # ── Name normalisation ────────────────────────────────────────────────
    def _normalise_name(raw_name):
        import unicodedata as _ud
        candidates = []
        def _add(s):
            s = s.strip()
            if s and s not in candidates:
                candidates.append(s)
        name = raw_name.strip()
        _add(name)
        if "," in name:
            _add(name.split(",")[0].strip())
        import re as _re2
        no_paren = _re2.sub(r"\s*\(.*?\)", "", name).strip()
        _add(no_paren)
        if "," in no_paren:
            _add(no_paren.split(",")[0].strip())
        for c in list(candidates):
            _add(c.title())
        for c in list(candidates):
            ascii_c = _ud.normalize("NFD", c)
            ascii_c = "".join(ch for ch in ascii_c
                              if _ud.category(ch) != "Mn")
            _add(ascii_c)
        return candidates

    # ── Country QID lookup cache ──────────────────────────────────────────
    _COUNTRY_QID_CACHE = {}

    def _get_country_qid(country_name):
        if country_name in _COUNTRY_QID_CACHE:
            return _COUNTRY_QID_CACHE[country_name]
        try:
            r = _req.get("https://www.wikidata.org/w/api.php", params={
                "action": "wbsearchentities", "search": country_name,
                "language": "en", "type": "item", "limit": 5, "format": "json",
            }, headers={"User-Agent": "GeoQueryCA-validator/1.0"}, timeout=8)
            r.raise_for_status()
            for item in r.json().get("search", []):
                desc = item.get("description", "").lower()
                if any(w in desc for w in ["country","sovereign state","republic",
                                           "kingdom","nation","federal"]):
                    _COUNTRY_QID_CACHE[country_name] = item["id"]
                    return item["id"]
        except Exception:
            pass
        _COUNTRY_QID_CACHE[country_name] = None
        return None

    _CATEGORY_P31 = {
        "Religious Site":    ["Q34627","Q16970","Q44539","Q317557"],
        "Temple":            ["Q44539","Q622425"],
        "Mosque":            ["Q32815"],
        "Church":            ["Q16970","Q2977"],
        "Monument":          ["Q4989906","Q179700"],
        "Palace":            ["Q16560","Q483110"],
        "Fort":              ["Q57821","Q1785071"],
        "Museum":            ["Q33506"],
        "Historical Site":   ["Q839954","Q4989906"],
        "Natural Attraction":["Q4421","Q23442"],
        "University":        ["Q3918"],
        "Park":              ["Q22698"],
    }

    def _fetch_wikidata_coords(monument_name, country, category=""):
        url  = "https://query.wikidata.org/sparql"
        hdrs = {"User-Agent": "GeoQueryCA-validator/1.0",
                "Accept":     "application/sparql-results+json"}
        candidates   = _normalise_name(monument_name)
        country_qid  = _get_country_qid(country) if country else None
        p31_qids     = _CATEGORY_P31.get(category, [])

        def _run(sparql):
            try:
                r = _req.get(url, params={"query": sparql, "format": "json"},
                             headers=hdrs, timeout=15)
                r.raise_for_status()
                out = []
                for b in r.json()["results"]["bindings"]:
                    try:
                        out.append((float(b["lat"]["value"]),
                                    float(b["lon"]["value"])))
                    except Exception:
                        pass
                return out
            except Exception:
                return []

        for cand in candidates[:4]:
            safe = cand.replace('"', '\\"')
            if country_qid:
                hits = _run(f"""SELECT ?lat ?lon WHERE {{
                  ?item rdfs:label "{safe}"@en .
                  ?item wdt:P17 wd:{country_qid} .
                  ?item wdt:P625 ?coord .
                  BIND(geof:latitude(?coord) AS ?lat)
                  BIND(geof:longitude(?coord) AS ?lon)
                }} LIMIT 5""")
                if hits: return hits[0]
                _time.sleep(0.25)
            for qid in p31_qids[:2]:
                hits = _run(f"""SELECT ?lat ?lon WHERE {{
                  ?item rdfs:label "{safe}"@en .
                  ?item wdt:P31 wd:{qid} .
                  ?item wdt:P625 ?coord .
                  BIND(geof:latitude(?coord) AS ?lat)
                  BIND(geof:longitude(?coord) AS ?lon)
                }} LIMIT 5""")
                if hits: return hits[0]
            _time.sleep(0.25)
        for cand in candidates[:3]:
            safe = cand.replace('"', '\\"')
            hits = _run(f"""SELECT ?lat ?lon WHERE {{
              ?item rdfs:label "{safe}"@en .
              ?item wdt:P625 ?coord .
              BIND(geof:latitude(?coord) AS ?lat)
              BIND(geof:longitude(?coord) AS ?lon)
            }} LIMIT 5""")
            if hits: return hits[0]
            _time.sleep(0.25)
        return (None, None)

    def _fetch_wikipedia_coords(monument_name, city="", country="",
                                ds_lat=None, ds_lon=None):
        MAX_KM  = 500
        url_en  = "https://en.wikipedia.org/w/api.php"
        ua      = {"User-Agent": "GeoQueryCA-validator/1.0"}
        cands   = _normalise_name(monument_name)
        if city and city.lower() not in monument_name.lower():
            for c in cands[:2]: cands.append(f"{c}, {city}")

        def _coords(api_url, title):
            try:
                r = _req.get(api_url, params={
                    "action": "query", "titles": title,
                    "prop": "coordinates", "format": "json", "redirects": 1,
                }, headers=ua, timeout=10)
                r.raise_for_status()
                for page in r.json().get("query",{}).get("pages",{}).values():
                    if page.get("coordinates"):
                        c = page["coordinates"][0]
                        return float(c["lat"]), float(c["lon"])
            except Exception:
                pass
            return None

        def _ok(lat, lon):
            if ds_lat is None: return True
            try:
                return float(haversine_km(
                    np.array([ds_lat]), np.array([ds_lon]),
                    np.array([lat]),    np.array([lon]))[0]) <= MAX_KM
            except Exception:
                return True

        for c in cands[:6]:
            res = _coords(url_en, c)
            if res and _ok(*res): return res
            _time.sleep(0.15)
        try:
            for c in cands[:3]:
                r = _req.get(url_en, params={
                    "action": "query", "list": "search",
                    "srsearch": c + (f" {country}" if country else ""),
                    "srnamespace": 0, "srlimit": 5, "format": "json",
                }, headers=ua, timeout=10)
                r.raise_for_status()
                for hit in r.json().get("query",{}).get("search",[]):
                    res = _coords(url_en, hit["title"])
                    if res and _ok(*res): return res
                _time.sleep(0.15)
        except Exception:
            pass
        _REGIONAL = {"India":["hi","te","ta","kn","ml","mr","bn","gu"],
                     "Pakistan":["ur"],"Bangladesh":["bn"],"default":[]}
        for lang in (_REGIONAL.get(country,[]) + _REGIONAL["default"])[:3]:
            wp = f"https://{lang}.wikipedia.org/w/api.php"
            for c in cands[:2]:
                res = _coords(wp, c)
                if res and _ok(*res): return res
            _time.sleep(0.15)
        return (None, None)

    def _fetch_ground_truth(monument_name, city, country,
                            category="", ds_lat=None, ds_lon=None):
        lat, lon = _fetch_wikipedia_coords(
            monument_name, city, country, ds_lat, ds_lon)
        if lat is not None:
            return lat, lon, "Wikipedia"
        lat, lon = _fetch_wikidata_coords(monument_name, country, category)
        if lat is not None:
            return lat, lon, "Wikidata"
        return None, None, "not_found"

    # ── Deduplicated monument list ────────────────────────────────────────
    val_cols = [c for c in ["Monument","City","Country","Category",
                             "Lat_DD","Lon_DD"] if c in df.columns]
    monuments_df = (df[val_cols]
                    .dropna(subset=["Lat_DD","Lon_DD"])
                    .drop_duplicates(subset=["Monument"])
                    .reset_index(drop=True))
    N_val = len(monuments_df)
    print(f"Validating {N_val:,} unique monuments against Wikipedia / Wikidata …")
    print(f"(~2–4 s per monument — estimated total: "
          f"{N_val*3//60}–{N_val*5//60} min)")

    # Load cache
    _cache_csv = os.path.join(COORD_VAL_DIR, "ground_truth_coords_cache.csv")
    cached = pd.read_csv(_cache_csv) if os.path.exists(_cache_csv) \
             else pd.DataFrame(columns=["Monument","gt_lat","gt_lon","source"])
    if len(cached):
        print(f"  [cache] loaded {len(cached):,} previously fetched results.")

    def _safe_float(v):
        try:
            f = float(v)
            return None if (f != f) else f
        except (TypeError, ValueError):
            return None

    cached_index = {}
    for _, row in cached.iterrows():
        cached_index[str(row["Monument"])] = (
            _safe_float(row["gt_lat"]),
            _safe_float(row["gt_lon"]),
            str(row.get("source","not_found")))

    _refetch = {n for n,(la,lo,s) in cached_index.items() if s=="not_found"}
    if _refetch:
        print(f"  [cache] {len(_refetch):,} previously 'not_found' entries "
              f"will be re-fetched with improved fetchers.")
        for n in _refetch: del cached_index[n]

    # ── Main fetch loop ───────────────────────────────────────────────────
    results = []
    for _, row in tqdm(monuments_df.iterrows(), total=N_val,
                       desc="Fetching ground-truth coords"):
        name     = str(row["Monument"])
        country  = str(row.get("Country",""))
        city     = str(row.get("City",""))
        category = str(row.get("Category",""))
        ds_lat   = float(row["Lat_DD"])
        ds_lon   = float(row["Lon_DD"])

        if name in cached_index:
            gt_lat, gt_lon, source = cached_index[name]
            if gt_lat is None or gt_lon is None:
                source = "not_found"
        else:
            gt_lat, gt_lon, source = _fetch_ground_truth(
                name, city, country, category=category,
                ds_lat=ds_lat, ds_lon=ds_lon)
            _time.sleep(0.5)

        err_km = None
        if gt_lat is not None and gt_lon is not None:
            try:
                err_km = float(haversine_km(
                    np.array([ds_lat]), np.array([ds_lon]),
                    np.array([gt_lat]), np.array([gt_lon]))[0])
            except Exception:
                pass

        results.append({
            "Monument":  name,
            "Country":   country,
            "City":      city,
            "Category":  category,
            "ds_lat":    ds_lat,
            "ds_lon":    ds_lon,
            "gt_lat":    gt_lat,
            "gt_lon":    gt_lon,
            "source":    source,
            "error_km":  err_km,
        })

    val_df = pd.DataFrame(results)

    # Save cache
    new_cache = pd.DataFrame([
        {"Monument": r["Monument"], "gt_lat": r["gt_lat"],
         "gt_lon": r["gt_lon"], "source": r["source"]}
        for r in results])
    combined = pd.concat([new_cache, cached], ignore_index=True)\
                 .drop_duplicates(subset=["Monument"], keep="first")
    combined.to_csv(_cache_csv, index=False)

    # ── Statistics ───────────────────────────────────────────────────────
    _real = {"Wikipedia","Wikidata"}
    found    = val_df[val_df["source"].isin(_real) &
                      val_df["gt_lat"].notna() &
                      val_df["gt_lon"].notna()].copy()
    notfound = val_df[~val_df["source"].isin(_real)]
    err      = found["error_km"].dropna()

    print(f"\n  ═══ Validation Results ═══")
    print(f"  Found : {len(found):,} / {N_val:,} ({len(found)/N_val*100:.1f}%)")
    print(f"  Source: Wikipedia={(found['source']=='Wikipedia').sum()}, "
          f"Wikidata={(found['source']=='Wikidata').sum()}")
    print(f"  Not found : {len(notfound):,}")
    if len(err):
        print(f"\n  Error vs ground truth (Wikipedia / Wikidata):")
        print(f"    MAE    : {err.mean():.3f} km")
        print(f"    Median : {err.median():.3f} km")
        print(f"    Std    : {err.std():.3f} km")
        print(f"    Max    : {err.max():.3f} km")
        print(f"    ≤ 1 km : {(err<=1).mean()*100:.1f}%  "
              f"≤ 10 km : {(err<=10).mean()*100:.1f}%  "
              f"≤ 50 km : {(err<=50).mean()*100:.1f}%  "
              f"> 100 km: {(err>100).mean()*100:.1f}%  "
              f"(likely wrong Wikipedia match)")

    # Worst offenders
    worst_df = found[found["error_km"].notna()]\
                   .nlargest(20, "error_km")[
        ["Monument","Country","City","Category",
         "ds_lat","ds_lon","gt_lat","gt_lon","source","error_km"]]
    worst_df.to_csv(os.path.join(COORD_VAL_DIR,"worst_coord_errors.csv"),
                    index=False)
    print(f"\n  Top-5 largest errors:")
    for _, r in worst_df.head(5).iterrows():
        print(f"    {str(r['Monument'])[:35]:<35}  "
              f"ds=({r['ds_lat']:.3f},{r['ds_lon']:.3f})  "
              f"gt=({r['gt_lat']:.3f},{r['gt_lon']:.3f})  "
              f"err={r['error_km']:.1f} km  [{r['source']}]")

    # ── Figures ───────────────────────────────────────────────────────────
    if len(err) >= 5:
        print("\n  Rendering validation figures …")
        fig, axes = plt.subplots(1, 2,
                                 figsize=(SN_2COL, SN_2COL*0.38), dpi=SN_DPI)
        fig.patch.set_facecolor(C_WHITE)

        ax = axes[0]
        _bins = np.logspace(np.log10(max(err.min(), 0.01)),
                            np.log10(err.max()+1), 30)
        _, _, patches_h = ax.hist(err, bins=_bins,
                                  color=C_BLUE2, edgecolor=C_WHITE,
                                  linewidth=0.3)
        for p, left in zip(patches_h, _bins[:-1]):
            if left <= 10:    p.set_facecolor(C_NAVY)
            elif left <= 100: p.set_facecolor(C_BLUE3)
            else:             p.set_facecolor(C_LBLUE)
        ax.set_xscale("log")
        ax.set_xlabel("Dataset vs Wikipedia/Wikidata error (km, log scale)")
        ax.set_ylabel("Monument count")
        ax.set_title("(a) Coordinate Error Distribution")
        ax.axvline(10,  color=C_RED,   lw=0.7, ls="--", alpha=0.7)
        ax.axvline(100, color=C_MGREY, lw=0.7, ls="--", alpha=0.7)
        ax.legend(handles=[
            mpatches.Patch(color=C_NAVY,  label="≤ 10 km (high precision)"),
            mpatches.Patch(color=C_BLUE3, label="10–100 km"),
            mpatches.Patch(color=C_LBLUE, label="> 100 km (suspect match)"),
        ], fontsize=5.5, frameon=True, edgecolor=C_LGREY, loc="upper right")

        ax = axes[1]
        _sorted = np.sort(err.values)
        _cdf    = np.arange(1, len(_sorted)+1) / len(_sorted)
        ax.plot(_sorted, _cdf*100, color=C_NAVY, lw=1.2)
        ax.set_xscale("log")
        for thresh, col in [(1,C_NAVY),(10,C_BLUE2),(50,C_BLUE3),(100,C_MGREY)]:
            pct = float((err<=thresh).mean()*100)
            ax.axvline(thresh, color=col, lw=0.7, ls="--", alpha=0.8)
            ax.text(thresh*1.15, pct+2, f"{pct:.0f}%",
                    fontsize=5.5, color=col)
        ax.set_xlabel("Error threshold (km, log scale)")
        ax.set_ylabel("Cumulative % of monuments")
        ax.set_title("(b) Cumulative Error Distribution (CDF)")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
        fig.suptitle(
            f"Coordinate Accuracy vs Wikipedia/Wikidata Ground Truth\n"
            f"(N={len(err):,} matched;  "
            f"MAE={err.mean():.2f} km,  Median={err.median():.2f} km)",
            fontsize=7, fontweight="bold", color=C_NAVY)
        plt.tight_layout(rect=[0,0,1,0.87])
        _save(fig, COORD_VAL_DIR, "fig_coord_validation")
        plt.show()

        # Spatial scatter
        fig2, ax2 = plt.subplots(
            figsize=(SN_15COL, SN_15COL*0.85), dpi=SN_DPI)
        fig2.patch.set_facecolor(C_WHITE)
        _ev = found["error_km"].fillna(0).values
        _vm = float(np.percentile(_ev[_ev>0], 95)) if (_ev>0).any() else 1
        sc  = ax2.scatter(found["ds_lon"], found["ds_lat"],
                          c=_ev, cmap="Blues", vmin=0, vmax=_vm,
                          s=6, alpha=0.7, edgecolors="none")
        cb  = plt.colorbar(sc, ax=ax2, shrink=0.7)
        cb.set_label("Error vs GT (km)", fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
        for _, r in worst_df.head(10).iterrows():
            if pd.notna(r["gt_lat"]) and pd.notna(r["gt_lon"]):
                ax2.plot([r["ds_lon"], float(r["gt_lon"])],
                         [r["ds_lat"], float(r["gt_lat"])],
                         color=C_RED, lw=0.5, alpha=0.6)
                ax2.scatter(float(r["gt_lon"]), float(r["gt_lat"]),
                            marker="x", color=C_RED, s=12, lw=0.8)
        ax2.axhline(0, color=C_MGREY, lw=0.4, ls="--")
        ax2.set_xlabel("Longitude (°)"); ax2.set_ylabel("Latitude (°)")
        ax2.set_title(
            "Spatial Distribution of Coordinate Errors\n"
            "(blue = dataset coords coloured by error; "
            "red lines = top-10 largest discrepancies)",
            fontsize=6.5, color=C_NAVY)
        plt.tight_layout()
        _save(fig2, COORD_VAL_DIR, "fig_coord_validation_map")
        plt.show()

    # ── JSON summary ──────────────────────────────────────────────────────
    cv_metrics = {"n_unique_monuments": N_val,
                  "n_found":  int(len(found)),
                  "n_not_found": int(len(notfound)),
                  "found_pct": float(len(found)/N_val*100),
                  "source_wikipedia": int((found["source"]=="Wikipedia").sum()),
                  "source_wikidata":  int((found["source"]=="Wikidata").sum())}
    if len(err):
        cv_metrics.update({
            "MAE_km":          float(err.mean()),
            "median_km":       float(err.median()),
            "std_km":          float(err.std()),
            "max_km":          float(err.max()),
            "pct_within_1km":  float((err<=1).mean()*100),
            "pct_within_10km": float((err<=10).mean()*100),
            "pct_within_50km": float((err<=50).mean()*100),
            "pct_over_100km":  float((err>100).mean()*100),
        })
    with open(os.path.join(COORD_VAL_DIR,
                           "coord_validation_metrics.json"), "w") as f:
        json.dump(_jsonsafe(cv_metrics), f, indent=2)

    val_df.to_csv(os.path.join(COORD_VAL_DIR,
                               "coordinate_validation.csv"), index=False)
    coord_validation_summary = cv_metrics
    print(f"\nPhase 1.5 outputs → {COORD_VAL_DIR}/")
    print("  coordinate_validation.csv     — full per-monument results")
    print("  worst_coord_errors.csv        — top-20 largest errors")
    print("  coord_validation_metrics.json")
    print("  fig_coord_validation          — histogram + CDF")
    print("  fig_coord_validation_map      — spatial scatter")

elif CONFIG.get("run_coord_validation") and not HAS_COORDS:
    print("\nPhase 1.5 skipped — no valid coordinate columns in dataset.")

if CONFIG["run_benchmark"]:
    print("\n" + "═"*72 + "\nPHASE 2 — Zero-shot Retrieval Benchmark\n" + "═"*72)

    queries  = df["Contextual_Query"].tolist()
    passages = df["Cultural_Significance"].tolist()
    langs    = df["Query_Language"].tolist()
    groups_arr = df["group"].to_numpy()

    # ── 4.1 Metric primitives ────────────────────────────────────────────
    def ranks_from_scores(scores, target_cols=None):
        # Tie-aware ranking, PESSIMISTIC within a tie group.
        #   rank = 1 + (# strictly greater) + (# equal, excluding the target)
        # i.e. the target is placed at the BOTTOM of everything it ties with.
        # This is essential for BM25 on cross-script queries: when a query
        # shares no tokens with any passage, BM25 scores the entire pool 0, so
        # the target ties with all N passages. A naive "1 + (# strictly greater)"
        # would call that rank 1 (a perfect hit) on queries BM25 actually failed,
        # massively inflating its scores. Counting tied passages against the
        # target gives it rank N in that case — the correct worst-case outcome.
        Nq = scores.shape[0]
        if target_cols is None:
            target_cols = torch.arange(Nq, device=scores.device)
        target_scores = scores.gather(1, target_cols.unsqueeze(1))   # (Nq, 1)
        strictly_greater = (scores > target_scores).sum(dim=1)
        tied             = (scores == target_scores).sum(dim=1) - 1   # exclude self
        ranks = strictly_greater + tied + 1
        return ranks.cpu().numpy().astype(np.int64)

    def per_query_metric_arrays(ranks):
        r = ranks.astype(np.float64)
        out = {"MRR": 1.0 / r}
        for k in CONFIG["k_values"]:
            out[f"Recall@{k}"] = (ranks <= k).astype(np.float64)
        for k in CONFIG["ndcg_k_values"]:
            out[f"nDCG@{k}"] = (ranks <= k).astype(np.float64) / np.log2(r + 1)
        return out

    def bootstrap_ci(values, B, alpha, seed):
        rng  = np.random.default_rng(seed)
        n    = len(values)
        idx  = rng.integers(0, n, size=(B, n))
        boot = values[idx].mean(axis=1)
        return (float(values.mean()),
                float(np.percentile(boot, 100 * alpha / 2)),
                float(np.percentile(boot, 100 * (1 - alpha / 2))),
                float(boot.std(ddof=1)))

    def paired_bootstrap_pvalue(va, vb, B, seed):
        rng = np.random.default_rng(seed)
        diff = va - vb
        n = len(diff); obs = diff.mean()
        if obs == 0: return 1.0
        idx = rng.integers(0, n, size=(B, n))
        boot_mean = diff[idx].mean(axis=1)
        centred = boot_mean - obs
        return float(min((np.sum(np.abs(centred) >= np.abs(obs)) + 1) / (B + 1),
                         1.0))

    def holm_bonferroni(p_values, alpha=0.05):
        p = np.asarray(p_values, dtype=float); m = len(p); order = np.argsort(p)
        adj = np.empty(m); run = 0.0
        for i, j in enumerate(order):
            run = max(run, (m - i) * p[j])
            adj[j] = min(run, 1.0)
        return adj

    def metrics_with_ci(per_q):
        out = {}
        for k, v in per_q.items():
            m, lo, hi, se = bootstrap_ci(v, CONFIG["bootstrap_B"],
                                         CONFIG["bootstrap_alpha"],
                                         CONFIG["random_seed"])
            out[k] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "se": se}
        return out

    # ── 4.2 Encoders ─────────────────────────────────────────────────────
    def _cache_path(model_name, kind, extra=""):
        key = hashlib.md5(
            f"{model_name}|{N}|{CONFIG['max_seq_length']}|{extra}".encode()
        ).hexdigest()[:10]
        return os.path.join(CACHE_DIR, f"{kind}_{key}.pt")

    def encode_hf(texts, model_name, kind, prefix="", force_cpu=False):
        dev = torch.device("cpu") if force_cpu else DEVICE
        if prefix: texts = [prefix + t for t in texts]
        cache = _cache_path(model_name, kind, prefix)
        if os.path.exists(cache):
            print(f"  [cache] {os.path.basename(cache)}")
            return torch.load(cache, map_location=dev)
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
        def mean_pool(out, mask):
            toks = out[0]
            m = mask.unsqueeze(-1).expand(toks.size()).float()
            return (toks * m).sum(1) / m.sum(1).clamp(min=1e-9)
        embs = []; bs = CONFIG["batch_size"]
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), bs),
                          desc=f"encode {model_name.split('/')[-1]} [{kind}]"):
                enc = tok(texts[i:i+bs], padding=True, truncation=True,
                          max_length=CONFIG["max_seq_length"],
                          return_tensors="pt").to(dev)
                out = mdl(**enc)
                emb = F.normalize(mean_pool(out, enc["attention_mask"]),
                                  p=2, dim=1)
                embs.append(emb)
        emb_all = torch.cat(embs, dim=0)
        torch.save(emb_all, cache)
        del mdl, tok
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        return emb_all

    def encode_sbert(texts, model_name, kind, prefix="", force_cpu=False):
        dev = torch.device("cpu") if force_cpu else DEVICE
        if prefix: texts = [prefix + t for t in texts]
        cache = _cache_path(model_name, kind, prefix)
        if os.path.exists(cache):
            print(f"  [cache] {os.path.basename(cache)}")
            return torch.load(cache, map_location=dev)
        from sentence_transformers import SentenceTransformer
        mdl = SentenceTransformer(model_name, device=str(dev),
                                  trust_remote_code=True)
        emb = mdl.encode(texts, batch_size=CONFIG["batch_size"],
                         convert_to_tensor=True, show_progress_bar=True,
                         normalize_embeddings=True).to(dev)
        torch.save(emb, cache)
        del mdl
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        return emb

    def score_matrix(mc, force_cpu=False):
        label, kind, name = mc["label"], mc["kind"], mc["name"]
        prefix_q = mc.get("prefix_q", "")
        prefix_p = mc.get("prefix_p", "")
        dev = torch.device("cpu") if force_cpu else DEVICE
        if kind == "bm25":
            from rank_bm25 import BM25Okapi
            tok = lambda t: re.findall(r"\w+", t.lower(), flags=re.UNICODE)
            bm25 = BM25Okapi([tok(p) for p in passages])
            S = np.zeros((N, N), dtype=np.float32)
            for i in tqdm(range(N), desc=f"BM25 scoring [{label}]"):
                S[i] = bm25.get_scores(tok(queries[i]))
            return torch.from_numpy(S).to(dev)
        if kind == "hf":
            q_emb = encode_hf(queries,  name, "q", prefix=prefix_q, force_cpu=force_cpu)
            p_emb = encode_hf(passages, name, "p", prefix=prefix_p, force_cpu=force_cpu)
        else:
            q_emb = encode_sbert(queries,  name, "q", prefix=prefix_q, force_cpu=force_cpu)
            p_emb = encode_sbert(passages, name, "p", prefix=prefix_p, force_cpu=force_cpu)
        return (q_emb @ p_emb.T).to(dev)

    # ── 4.3 Run all models ────────────────────────────────────────────────
    print("\nRunning models")
    per_query_ranks   = {}
    per_query_metrics = {}
    failed_models     = []
    geo_results_phase2 = {}   # label → geo-error dict (needs score matrix)
    for i, mc in enumerate(CONFIG["models"], 1):
        label = mc["label"]
        print(f"\n[{i}/{len(CONFIG['models'])}] {label}")
        S = None
        try:
            S = score_matrix(mc)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            estr = str(e).lower()
            # A device-side assert is UNRECOVERABLE: it leaves the CUDA context
            # in a sticky error state, so every later GPU op — in this phase and
            # every phase after — will re-raise the same assert. Swallowing it
            # and continuing (or retrying on CPU) only hides the corruption until
            # something innocent crashes downstream. Halt immediately instead.
            is_fatal_assert = ("device-side assert" in estr
                               or "cudaerrorassert" in estr
                               or "an illegal memory access" in estr)
            if is_fatal_assert and DEVICE.type == "cuda":
                print(f"\n  ✗  FATAL: {label} triggered a CUDA device-side assert.")
                print(f"     {err}")
                print( "     The CUDA context is now corrupted and cannot be reused;")
                print( "     continuing would crash a later phase with a misleading")
                print( "     traceback. Stopping here.")
                print( "     → Remove or replace this model in CONFIG['models'],")
                print( "       then RESTART the runtime (a restart is required to")
                print( "       clear the poisoned context) and re-run.")
                raise RuntimeError(
                    f"Unrecoverable CUDA device-side assert from model "
                    f"'{label}'. Remove it from CONFIG['models'] and restart "
                    f"the runtime.") from e
            is_cuda = ("CUDA" in str(e) or "CUBLAS" in str(e)
                       or "cublas" in str(e) or "out of memory" in str(e).lower())
            print(f"  ⚠  {label} failed on {DEVICE.type.upper()}: {err}")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            if is_cuda and DEVICE.type == "cuda":
                print(f"  ↻  Retrying {label} on CPU (slower, but reliable)…")
                try:
                    S = score_matrix(mc, force_cpu=True)
                    print(f"  ✓  {label} succeeded on CPU.")
                except Exception as e2:
                    print(f"  ⚠  {label} also failed on CPU: "
                          f"{type(e2).__name__}: {str(e2).splitlines()[0][:160]}")
                    S = None
            if S is None:
                print(f"  ⏭  Skipping {label}; benchmark continues with the rest.")
                failed_models.append(label)
                continue
        ranks = ranks_from_scores(S)
        per_query_ranks[label] = ranks
        per_query_metrics[label] = per_query_metric_arrays(ranks)

        # ── Geo-error metrics (coordinate-aware, uses full score matrix) ──
        if HAS_COORDS:
            try:
                S_np = S.cpu().numpy().astype(np.float32)
                geo_results_phase2[label] = geo_error_metrics_from_score_matrix(
                    S_np, LAT_ARR, LON_ARR, ranks)
                print(f"  Geo: MAE={geo_results_phase2[label]['MAE_geo_km']:.1f} km  "
                      f"GeoAcc@161km={geo_results_phase2[label].get('GeoAcc@161km',0)*100:.1f}%  "
                      f"geo-wMRR={geo_results_phase2[label].get('geo_weighted_MRR',float('nan')):.4f}")
            except Exception as _ge:
                print(f"  ⚠  Geo-error metrics failed for {label}: {_ge}")
                geo_results_phase2[label] = {}
        # in-pool ranks per group
        in_pool = {}
        for g in GROUPS_ORDER:
            idx = np.where(groups_arr == g)[0]
            if len(idx) == 0:
                in_pool[g] = np.empty(0, dtype=np.int64); continue
            idx_t = torch.from_numpy(idx).to(S.device)
            sub = S.index_select(0, idx_t).index_select(1, idx_t)
            tgt = torch.arange(len(idx), device=S.device)
            in_pool[g] = ranks_from_scores(sub, tgt)
        per_query_ranks[f"{label}__inpool"] = in_pool
        del S
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    if failed_models:
        print(f"\n⚠  {len(failed_models)} model(s) skipped: {failed_models}")
        print("   Results below cover only the models that completed. "
              "See troubleshooting note at end of run.")

    np.savez(os.path.join(BENCH_DIR, "per_query_ranks.npz"),
             **{k: v for k, v in per_query_ranks.items()
                if isinstance(v, np.ndarray)})

    # ── 4.4 Aggregate metrics ─────────────────────────────────────────────
    # Only keep models that actually produced results.
    LABELS  = [m["label"] for m in CONFIG["models"]
               if m["label"] not in failed_models]
    METRICS = ["MRR"] + [f"nDCG@{k}" for k in CONFIG["ndcg_k_values"]] \
                      + [f"Recall@{k}" for k in CONFIG["k_values"]]
    results = {"config": CONFIG, "N": N,
               "timestamp": datetime.utcnow().isoformat(),
               "overall": {}, "groups_inpool": {}, "groups_global": {},
               "per_language": {}, "geo_error": {}}
    for lab in LABELS:
        results["overall"][lab] = metrics_with_ci(per_query_metrics[lab])
    for lab in LABELS:
        results["groups_inpool"][lab] = {}
        for g in GROUPS_ORDER:
            ranks_g = per_query_ranks[f"{lab}__inpool"][g]
            if len(ranks_g) == 0:
                results["groups_inpool"][lab][g] = {"N": 0}; continue
            pq = per_query_metric_arrays(ranks_g)
            results["groups_inpool"][lab][g] = {"N": int(len(ranks_g)),
                                                **metrics_with_ci(pq)}
    for lab in LABELS:
        results["groups_global"][lab] = {}
        for g in GROUPS_ORDER:
            idx = np.where(groups_arr == g)[0]
            if len(idx) == 0:
                results["groups_global"][lab][g] = {"N": 0}; continue
            pq = {k: v[idx] for k, v in per_query_metrics[lab].items()}
            results["groups_global"][lab][g] = {"N": int(len(idx)),
                                                **metrics_with_ci(pq)}
    lang_counts = df["Query_Language"].value_counts()
    REPORTED_LANGS = lang_counts[lang_counts >= CONFIG["min_language_n"]].index.tolist()
    langs_arr = np.array(langs)
    for lab in LABELS:
        results["per_language"][lab] = {}
        for L in REPORTED_LANGS:
            idx = np.where(langs_arr == L)[0]
            pq  = {k: v[idx] for k, v in per_query_metrics[lab].items()}
            results["per_language"][lab][L] = {"N": int(len(idx)),
                                               **metrics_with_ci(pq)}

    # ── 4.4b Geo-error results ──────────────────────────────────────────
    if HAS_COORDS and geo_results_phase2:
        for lab in LABELS:
            results["geo_error"][lab] = geo_results_phase2.get(lab, {})

    # ── 4.5 Significance tests ────────────────────────────────────────────
    print("\nPairwise paired-bootstrap significance tests (MRR)…")
    pairs, p_raw = [], []
    for i, A in enumerate(LABELS):
        for B_ in LABELS[i+1:]:
            p = paired_bootstrap_pvalue(per_query_metrics[A]["MRR"],
                                        per_query_metrics[B_]["MRR"],
                                        CONFIG["bootstrap_B"],
                                        CONFIG["random_seed"])
            pairs.append((A, B_)); p_raw.append(p)
    p_adj = holm_bonferroni(p_raw, CONFIG["bootstrap_alpha"]) if p_raw else []
    sig_table = []
    for (A, B_), pr, pa in zip(pairs, p_raw, p_adj):
        dm = per_query_metrics[A]["MRR"].mean() - per_query_metrics[B_]["MRR"].mean()
        sig_table.append({"A": A, "B": B_, "diff": float(dm),
                          "p_raw": float(pr), "p_holm": float(pa),
                          "significant": bool(pa < CONFIG["bootstrap_alpha"])})
    results["significance_MRR"] = sig_table

    print(f"{'Comparison':<22}{'Δ MRR':>10}{'p (raw)':>12}{'p (Holm)':>12}{'sig':>6}")
    for s in sig_table:
        star = "✓" if s["significant"] else ""
        print(f"{s['A']+' vs '+s['B']:<22}{s['diff']:>+10.4f}"
              f"{s['p_raw']:>12.4f}{s['p_holm']:>12.4f}{star:>6}")

    # ── 4.6 Persist metrics ──────────────────────────────────────────────
    with open(os.path.join(BENCH_DIR, "metrics.json"), "w") as f:
        json.dump(_jsonsafe(results), f, indent=2)
    rows = []
    for lab in LABELS:
        row = {"Model": lab, "N": N}
        for m in METRICS:
            r = results["overall"][lab][m]
            row[m] = r["mean"]
            row[f"{m}_ci_lo"] = r["ci_lo"]
            row[f"{m}_ci_hi"] = r["ci_hi"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(BENCH_DIR, "overall_metrics.csv"),
                              index=False)

    # ── 4.6.5 Table 2 — Model characteristics (paper Methods table) ──────
    print("\n[Render] Table 2 (model characteristics, zero-shot)")
    render_table2(LABELS, BENCH_DIR)

    # ── 4.7 Table 3 — overall with CIs ───────────────────────────────────
    print("\n[Render] Table 3 (overall with CIs)")
    def _fmt_ci(d): return f"{d['mean']:.4f}\n({d['ci_lo']:.4f}, {d['ci_hi']:.4f})"
    t3_cols = ["Model"] + METRICS
    t3_data = []
    for lab in LABELS:
        o = results["overall"][lab]
        display = f"{lab} (Lexical)" if lab == "BM25" else lab
        t3_data.append([display] + [_fmt_ci(o[m]) for m in METRICS])
    means = np.array([[results["overall"][lab][m]["mean"] for m in METRICS]
                      for lab in LABELS])
    best_row_for_col = means.argmax(axis=0) + 1

    fig, ax = plt.subplots(figsize=(SN_2COL, 0.22 * len(LABELS) + 0.9))
    fig.patch.set_facecolor(WHITE); ax.axis("off")
    table = ax.table(cellText=t3_data, colLabels=t3_cols, cellLoc="center",
                     loc="center",
                     colWidths=[0.16] + [0.14] * len(METRICS))
    table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.8)
    for j in range(len(t3_cols)):
        c = table[0, j]; c.set_facecolor(C_NAVY)
        c.set_text_props(color=TEXT_LIGHT, fontweight="bold")
        c.set_edgecolor(WHITE); c.set_linewidth(0.5)
    for i in range(1, len(t3_data) + 1):
        bg = C_VBLUE if i % 2 else BLUE_ALT
        for j in range(len(t3_cols)):
            c = table[i, j]; c.set_facecolor(bg)
            c.set_edgecolor(WHITE); c.set_linewidth(0.4)
            if j == 0:
                c.set_text_props(fontweight="bold", color=C_NAVY)
            elif j >= 1 and best_row_for_col[j - 1] == i:
                c.set_facecolor(HIGHLIGHT)
                c.set_text_props(fontweight="bold", color=C_NAVY)
            else:
                c.set_text_props(color=TEXT_DARK)
    ax.set_title(f"Table 3   Overall zero-shot benchmark   (N = {N:,})",
                 fontsize=8, fontweight="bold", color=C_NAVY, pad=6, loc="left")
    ax.text(0, 0.965,
            f"Mean (95% bootstrap CI, B={CONFIG['bootstrap_B']}). "
            "Best per metric highlighted.",
            transform=ax.transAxes, fontsize=6, color=C_MGREY, style="italic")
    plt.tight_layout(); _save(fig, BENCH_DIR, "table3_overall"); plt.show()

    # ── 4.8 Table 4 — cross-lingual breakdown (in-pool) ─────────────────
    print("[Render] Table 4 (cross-lingual, in-pool)")
    t4_cols = ["Model", "Language Group", "N", "MRR", "Recall@1", "Recall@5"]
    t4_data = []
    for lab in LABELS:
        for g in GROUPS_ORDER:
            gm = results["groups_inpool"][lab][g]
            def _f(key):
                if "mean" not in gm.get(key, {}): return "—"
                return f"{gm[key]['mean']:.4f}"
            if gm.get("N", 0) == 0:
                t4_data.append([lab, g, "0", "—", "—", "—"])
            else:
                t4_data.append([lab, g, f"{gm['N']:,}",
                                _f("MRR"), _f("Recall@1"), _f("Recall@5")])

    fig, ax = plt.subplots(figsize=(SN_2COL, max(SN_2COL * 0.55, 0.165 * len(t4_data) + 0.7)))
    fig.patch.set_facecolor(WHITE); ax.axis("off")
    table = ax.table(cellText=t4_data, colLabels=t4_cols, cellLoc="center",
                     loc="center",
                     colWidths=[0.13, 0.22, 0.09, 0.14, 0.14, 0.14])
    table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.45)
    for j in range(len(t4_cols)):
        c = table[0, j]; c.set_facecolor(C_NAVY)
        c.set_text_props(color=TEXT_LIGHT, fontweight="bold")
        c.set_edgecolor(WHITE); c.set_linewidth(0.5)
    n_groups = len(GROUPS_ORDER)
    block_colors = [C_VBLUE, BLUE_ALT, C_VBLUE, BLUE_ALT]
    indian_pos = GROUPS_ORDER.index("Indian vernaculars") \
                 if "Indian vernaculars" in GROUPS_ORDER else -1
    for i in range(1, len(t4_data) + 1):
        block = (i - 1) // n_groups
        row_within = (i - 1) % n_groups
        is_indian = (row_within == indian_pos)
        bg = WARN_BG if is_indian else block_colors[block % len(block_colors)]
        for j in range(len(t4_cols)):
            c = table[i, j]; c.set_facecolor(bg)
            c.set_edgecolor(WHITE); c.set_linewidth(0.4)
            if is_indian: c.set_text_props(fontweight="bold", color=WARN_FG)
            elif j == 0:  c.set_text_props(fontweight="bold", color=C_NAVY)
            else:         c.set_text_props(color=TEXT_DARK)
    for sep in range(n_groups, len(t4_data), n_groups):
        for j in range(len(t4_cols)):
            table[sep, j].set_edgecolor(C_MGREY)
            table[sep, j].set_linewidth(0.6)
    ax.set_title("Table 4   Cross-lingual retrieval (in-pool, by language group)",
                 fontsize=8, fontweight="bold", color=C_NAVY, pad=6, loc="left")
    ax.text(0, 0.96,
            "Each query ranked within its own language group's passage pool. "
            "Red rows: Indian vernaculars.",
            transform=ax.transAxes, fontsize=6, color=C_MGREY, style="italic")
    ax.legend(handles=[mpatches.Patch(facecolor=WARN_BG, edgecolor=WARN_FG,
                                      linewidth=0.6,
                                      label="Indian vernaculars")],
              loc="lower right", fontsize=6, framealpha=0.7,
              bbox_to_anchor=(1, -0.02))
    plt.tight_layout(); _save(fig, BENCH_DIR, "table4_crosslingual_inpool"); plt.show()

    # ── 4.9 Table 5 — significance matrix ────────────────────────────────
    print("[Render] Table 5 (significance matrix)")
    n_m = len(LABELS)
    delta_mat = np.zeros((n_m, n_m)); p_mat = np.ones((n_m, n_m))
    for s in sig_table:
        i, j = LABELS.index(s["A"]), LABELS.index(s["B"])
        delta_mat[i, j] =  s["diff"];   delta_mat[j, i] = -s["diff"]
        p_mat[i, j]     =  s["p_holm"]; p_mat[j, i]     =  s["p_holm"]

    fig, ax = plt.subplots(figsize=(SN_15COL, SN_15COL * 0.85))
    fig.patch.set_facecolor(WHITE)
    vmax = np.abs(delta_mat).max() or 1e-3
    img = ax.imshow(delta_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(n_m):
        for j in range(n_m):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=8, color=C_MGREY)
            else:
                star = ""
                if p_mat[i,j] < 0.001: star = "***"
                elif p_mat[i,j] < 0.01: star = "**"
                elif p_mat[i,j] < 0.05: star = "*"
                txt = f"Δ={delta_mat[i,j]:+.3f}\np={p_mat[i,j]:.3f}{star}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white" if abs(delta_mat[i,j]) > vmax*0.4 else TEXT_DARK)
    ax.set_xticks(range(n_m)); ax.set_yticks(range(n_m))
    ax.set_xticklabels(LABELS, fontsize=7, fontweight="bold")
    ax.set_yticklabels(LABELS, fontsize=7, fontweight="bold")
    ax.set_title("Table 5   Pairwise MRR comparison\n"
                 "(Δ = row − column, Holm-corrected paired-bootstrap p-value)",
                 fontsize=8, fontweight="bold", color=C_NAVY, pad=8)
    cb = plt.colorbar(img, ax=ax, shrink=0.72)
    cb.set_label("Δ MRR (row − column)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.text(0, n_m + 0.35,
            f"Significance markers: * p<0.05, ** p<0.01, *** p<0.001. "
            f"Holm–Bonferroni over {len(sig_table)} comparisons. "
            f"B={CONFIG['bootstrap_B']}.",
            fontsize=5.5, color=C_MGREY, style="italic", transform=ax.transData)
    plt.tight_layout(); _save(fig, BENCH_DIR, "table5_significance_matrix"); plt.show()

    # ── 4.9b Table 7 — Geo-error metrics ─────────────────────────────────
    if HAS_COORDS and any(geo_results_phase2.get(l) for l in LABELS):
        print("[Render] Table 7 (geo-error metrics)")
        GEO_COLS = ["MAE_geo_km", "MedAE_geo_km", "RMSE_geo_km",
                    "GeoAcc@1km", "GeoAcc@10km", "GeoAcc@50km", "GeoAcc@161km",
                    "geo_weighted_MRR", "p90_error_km"]
        GEO_HEADERS = ["Model", "MAE (km)", "MedAE (km)", "RMSE (km)",
                       "GeoAcc\n@1km", "GeoAcc\n@10km", "GeoAcc\n@50km",
                       "GeoAcc\n@161km", "Geo-wMRR", "p90 (km)"]
        t7_data = []
        for lab in LABELS:
            g = results["geo_error"].get(lab, {})
            if not g or "MAE_geo_km" not in g:
                t7_data.append([lab] + ["—"] * (len(GEO_COLS))); continue
            row = [lab]
            for col in GEO_COLS:
                v = g.get(col, float("nan"))
                if col.startswith("GeoAcc"):
                    row.append(f"{v*100:.1f}%")
                elif col == "geo_weighted_MRR":
                    row.append(f"{v:.4f}")
                else:
                    row.append(f"{v:.1f}")
            t7_data.append(row)

        # Determine best per column (lower=better for km, higher=better for acc/MRR)
        higher_better = {c: c.startswith("GeoAcc") or c == "geo_weighted_MRR"
                         for c in GEO_COLS}
        raw_vals = []
        for lab in LABELS:
            g = results["geo_error"].get(lab, {})
            raw_vals.append([g.get(col, float("nan")) for col in GEO_COLS])
        raw_arr = np.array(raw_vals, dtype=float)
        best_geo = []
        for j, col in enumerate(GEO_COLS):
            col_vals = raw_arr[:, j]
            valid_rows = np.where(np.isfinite(col_vals))[0]
            if len(valid_rows) == 0:
                best_geo.append(-1); continue
            if higher_better[col]:
                best_geo.append(valid_rows[np.argmax(col_vals[valid_rows])])
            else:
                best_geo.append(valid_rows[np.argmin(col_vals[valid_rows])])

        fig, ax = plt.subplots(figsize=(SN_2COL, 0.20 * len(LABELS) + 0.9))
        fig.patch.set_facecolor(WHITE); ax.axis("off")
        table = ax.table(cellText=t7_data, colLabels=GEO_HEADERS,
                         cellLoc="center", loc="center",
                         colWidths=[0.10] + [0.09] * len(GEO_COLS))
        table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.7)
        for j in range(len(GEO_HEADERS)):
            c = table[0, j]; c.set_facecolor(C_NAVY)
            c.set_text_props(color=TEXT_LIGHT, fontweight="bold")
            c.set_edgecolor(WHITE); c.set_linewidth(0.5)
        for i in range(1, len(t7_data) + 1):
            bg = C_VBLUE if i % 2 else BLUE_ALT
            for j in range(len(GEO_HEADERS)):
                c = table[i, j]; c.set_facecolor(bg)
                c.set_edgecolor(WHITE); c.set_linewidth(0.4)
                if j == 0:
                    c.set_text_props(fontweight="bold", color=C_NAVY)
                elif j >= 1 and best_geo[j-1] == i - 1:
                    c.set_facecolor(HIGHLIGHT)
                    c.set_text_props(fontweight="bold", color=C_NAVY)
                else:
                    c.set_text_props(color=TEXT_DARK)
        ax.set_title(f"Table 7   Geo-error metrics (coordinate-aware)   (N = {N:,})",
                     fontsize=8, fontweight="bold", color=C_NAVY, pad=6, loc="left")
        ax.text(0, 0.96,
                "Distance = Haversine between retrieved doc coords and ground-truth coords. "
                "GeoAcc@k = fraction within k km. Geo-wMRR = MRR × exp(−dist/100 km). "
                "Best per column highlighted.",
                transform=ax.transAxes, fontsize=5.5, color=C_MGREY, style="italic")
        plt.tight_layout(); _save(fig, BENCH_DIR, "table7_geo_error"); plt.show()

        # LaTeX export for Table 7
        tex7 = [r"\begin{table}[t]", r"\centering",
                r"\caption{Coordinate-aware geo-error metrics on GeoQueryCA. "
                r"Distance metrics (km) computed via Haversine formula between "
                r"the top-1 retrieved document's coordinates and the query's "
                r"ground-truth coordinates. GeoAcc@$k$ = fraction of queries "
                r"where top-1 retrieval is within $k$ km. "
                r"Geo-wMRR = distance-penalised MRR with 100-km half-life weight.}",
                r"\label{tab:geo_error}", r"\small",
                r"\begin{tabular}{l" + "c" * len(GEO_COLS) + "}",
                r"\toprule",
                " & ".join(h.replace("\n", " ") for h in GEO_HEADERS) + r" \\",
                r"\midrule"]
        for i_lab, (lab, row) in enumerate(zip(LABELS, t7_data)):
            cells = []
            for j, v in enumerate(row):
                if j == 0:
                    cells.append(_tex_escape(v))
                elif best_geo[j-1] == i_lab and v != "—":
                    cells.append(r"\textbf{" + _tex_escape(v) + r"}")
                else:
                    cells.append(_tex_escape(v))
            tex7.append(" & ".join(cells) + r" \\")
        tex7 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        with open(os.path.join(BENCH_DIR, "table7.tex"), "w") as f:
            f.write("\n".join(tex7))

        # Geo-error figure (4-panel)
        print("[Render] Fig — Geo-error 4-panel")
        geo_error_figure(
            {l: results["geo_error"][l] for l in LABELS
             if "MAE_geo_km" in results["geo_error"].get(l, {})},
            BENCH_DIR, "fig_geo_error",
            title=f"Coordinate-Aware Error Metrics by Model  (N = {N:,})")
    else:
        print("[Render] Table 7 / Geo-error figure skipped (no coordinate data)")
    print("[Render] MRR bar (English vs Indian vernaculars)")
    def _get(lab, grp, key):
        return results["groups_inpool"][lab][grp].get(key, {}).get("mean", float("nan"))
    mrr_en  = [_get(lab, "English",           "MRR") for lab in LABELS]
    mrr_ind = [_get(lab, "Indian vernaculars","MRR") for lab in LABELS]
    err_en  = [[results["groups_inpool"][lab]["English"]["MRR"]["mean"] -
                results["groups_inpool"][lab]["English"]["MRR"]["ci_lo"]
                for lab in LABELS],
               [results["groups_inpool"][lab]["English"]["MRR"]["ci_hi"] -
                results["groups_inpool"][lab]["English"]["MRR"]["mean"]
                for lab in LABELS]]
    err_ind = [[results["groups_inpool"][lab]["Indian vernaculars"]["MRR"]["mean"] -
                results["groups_inpool"][lab]["Indian vernaculars"]["MRR"]["ci_lo"]
                for lab in LABELS],
               [results["groups_inpool"][lab]["Indian vernaculars"]["MRR"]["ci_hi"] -
                results["groups_inpool"][lab]["Indian vernaculars"]["MRR"]["mean"]
                for lab in LABELS]]

    x, w = np.arange(len(LABELS)), 0.32
    fig, ax = plt.subplots(figsize=(SN_2COL, SN_2COL * 0.42))
    fig.patch.set_facecolor(C_WHITE); ax.set_facecolor(C_WHITE)
    b1 = ax.bar(x - w/2, mrr_en,  width=w, color=C_NAVY, label="English",
                yerr=err_en, capsize=3, error_kw=dict(ecolor=C_NAVY, lw=0.8),
                zorder=3, edgecolor=WHITE, linewidth=0.5)
    b2 = ax.bar(x + w/2, mrr_ind, width=w, color=C_BLUE3,
                label="Indian vernaculars",
                yerr=err_ind, capsize=3, error_kw=dict(ecolor=C_BLUE3, lw=0.8),
                zorder=3, edgecolor=WHITE, linewidth=0.5)
    ymax = max(mrr_en + mrr_ind) if max(mrr_en + mrr_ind) > 0 else 0.05
    ax.set_ylim(0, ymax * 1.28)
    for bar, val in zip(b1, mrr_en):
        ax.text(bar.get_x() + bar.get_width()/2, val + ymax * 0.04,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=6, color=C_NAVY, fontweight="bold")
    for bar, val in zip(b2, mrr_ind):
        txt = f"{val:.4f}" if val < 0.1 else f"{val:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2, val + ymax * 0.04,
                txt, ha="center", va="bottom",
                fontsize=6, color=C_BLUE3, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=7,
                                         fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel("MRR (95% CI)")
    ax.set_title("MRR: English vs Indian Vernaculars  (in-pool, zero-shot)",
                 fontsize=8, fontweight="bold", color=C_NAVY, pad=6)
    ax.legend(fontsize=6, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); _save(fig, BENCH_DIR, "fig_mrr_groups_bar"); plt.show()

    # ── 4.11 Per-language heatmap ────────────────────────────────────────
    if REPORTED_LANGS:
        print(f"[Render] Per-language heatmap ({len(REPORTED_LANGS)} languages)")
        lang_order = sorted(REPORTED_LANGS,
                            key=lambda L: -results["per_language"][LABELS[0]][L]["MRR"]["mean"])
        M = np.array([[results["per_language"][lab][L]["MRR"]["mean"]
                       for lab in LABELS] for L in lang_order])
        Ns = [results["per_language"][LABELS[0]][L]["N"] for L in lang_order]
        fig, ax = plt.subplots(figsize=(SN_15COL, max(SN_15COL * 0.5, 0.18 * len(lang_order) + 0.8)))
        fig.patch.set_facecolor(WHITE)
        im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if (val < 0.3 or val > 0.85) else TEXT_DARK,
                        fontweight="bold" if val == M[i].max() else "normal")
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, fontsize=7, fontweight="bold")
        ax.set_yticks(range(len(lang_order)))
        ax.set_yticklabels([f"{L}  (n={n})" for L, n in zip(lang_order, Ns)],
                           fontsize=6)
        ax.set_title(f"Per-language MRR (global pool, N={N:,})",
                     fontsize=8, fontweight="bold", color=C_NAVY, pad=6)
        cb = plt.colorbar(im, ax=ax, shrink=0.65); cb.set_label("MRR", fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
        ax.text(0, len(lang_order) + 0.55,
                f"Languages with ≥{CONFIG['min_language_n']} samples. "
                "Bold = best model for that language.",
                fontsize=5.5, color=C_MGREY, style="italic")
        plt.tight_layout(); _save(fig, BENCH_DIR, "fig_per_language_heatmap"); plt.show()

    # ── 4.12 Bootstrap CI visualisation (replaces sparse forest plot) ────
    print("[Render] Bootstrap CI visualisation")

    # ── Colour map: one shade per model, dark→light by mean MRR rank ─────
    _mrr_order = sorted(LABELS,
                        key=lambda l: results["overall"][l]["MRR"]["mean"])
    _n = len(LABELS)
    _model_colors = {lab: PALETTE[i % len(PALETTE)]
                     for i, lab in enumerate(_mrr_order)}

    # ══════════════════════════════════════════════════════════════════════
    # Panel layout:
    #   Row 1 (full width) — connected dot plot: all metrics, models as
    #                        lines so trends across metrics are visible
    #   Row 2 left          — grouped bar chart for MRR + Recall@10
    #                         with shaded 95% CI rectangles
    #   Row 2 right         — radar / spider chart across all 6 metrics
    # ══════════════════════════════════════════════════════════════════════
    import matplotlib.gridspec as _gs2
    import matplotlib.patches as _mp2

    fig_bs = plt.figure(figsize=(SN_2COL, SN_2COL * 0.90), dpi=SN_DPI)
    fig_bs.patch.set_facecolor(C_WHITE)
    gs_bs = _gs2.GridSpec(2, 2, figure=fig_bs,
                          height_ratios=[1.0, 1.1],
                          hspace=0.52, wspace=0.28)
    ax_top  = fig_bs.add_subplot(gs_bs[0, :])   # full-width top
    ax_bar  = fig_bs.add_subplot(gs_bs[1, 0])   # bottom-left: CI bar
    ax_rad  = fig_bs.add_subplot(gs_bs[1, 1],   # bottom-right: radar
                                  polar=True)

    # ── (a) Connected dot plot — models as lines across all metrics ───────
    x_pos = np.arange(len(METRICS))
    for lab in LABELS:
        means = [results["overall"][lab][m]["mean"] for m in METRICS]
        lo    = [results["overall"][lab][m]["ci_lo"] for m in METRICS]
        hi    = [results["overall"][lab][m]["ci_hi"] for m in METRICS]
        col   = _model_colors[lab]

        # Shaded CI band
        ax_top.fill_between(x_pos, lo, hi, alpha=0.12, color=col)
        # Connected line
        ax_top.plot(x_pos, means, "o-", color=col, lw=1.2,
                    markersize=4, markeredgecolor=C_WHITE,
                    markeredgewidth=0.5, label=lab, zorder=3)

    ax_top.set_xticks(x_pos)
    ax_top.set_xticklabels(METRICS, fontsize=6.5, fontweight="bold")
    ax_top.set_ylabel("Score", fontsize=6)
    ax_top.set_ylim(0, None)
    ax_top.set_xlim(-0.3, len(METRICS) - 0.7)
    ax_top.set_title(
        "(a) All Metrics — Connected Dot Plot with 95% Bootstrap CI Bands",
        fontsize=7, fontweight="bold", color=C_NAVY, pad=4)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    # Legend to the right of the panel
    ax_top.legend(
        fontsize=5.5, loc="upper left",
        frameon=True, edgecolor=C_LGREY, framealpha=0.9,
        handlelength=1.2, labelspacing=0.3,
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    ax_top.yaxis.grid(True, color=C_LGREY, lw=0.35, alpha=0.6)
    ax_top.set_axisbelow(True)

    # ── (b) Grouped horizontal bar chart: MRR with explicit CI whiskers ───
    # Models sorted best→worst by MRR; CI shown as horizontal error bars
    _sorted_labs = sorted(LABELS,
                          key=lambda l: results["overall"][l]["MRR"]["mean"],
                          reverse=True)
    _bar_metrics = ["MRR", "Recall@10"]   # two most interpretable metrics
    _bar_w = 0.35
    y_pos = np.arange(len(_sorted_labs))

    for b_idx, bm in enumerate(_bar_metrics):
        _offset = (b_idx - 0.5) * _bar_w
        _means = [results["overall"][l][bm]["mean"]  for l in _sorted_labs]
        _lo    = [results["overall"][l][bm]["ci_lo"] for l in _sorted_labs]
        _hi    = [results["overall"][l][bm]["ci_hi"] for l in _sorted_labs]
        _xerr  = np.array([np.array(_means) - np.array(_lo),
                           np.array(_hi)    - np.array(_means)])
        _col   = C_NAVY if b_idx == 0 else C_BLUE3
        ax_bar.barh(y_pos + _offset, _means, height=_bar_w,
                    color=_col, edgecolor=C_WHITE, linewidth=0.3,
                    alpha=0.85, label=bm)
        ax_bar.errorbar(_means, y_pos + _offset,
                        xerr=_xerr, fmt="none",
                        ecolor=C_DGREY, elinewidth=0.8, capsize=2.5,
                        capthick=0.8, zorder=4)
        # Value labels
        for val, y_v in zip(_means, y_pos + _offset):
            ax_bar.text(val + max(_means) * 0.02, y_v,
                        f"{val:.3f}", va="center",
                        fontsize=5, color=C_DGREY)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(_sorted_labs, fontsize=6, fontweight="bold")
    ax_bar.set_xlabel("Score (with 95% bootstrap CI whiskers)", fontsize=6)
    ax_bar.set_title("(b) MRR and Recall@10  (ranked, best → worst)",
                     fontsize=7, fontweight="bold", color=C_NAVY, pad=4)
    ax_bar.set_xlim(0, max(
        results["overall"][l]["Recall@10"]["ci_hi"] for l in LABELS) * 1.25)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.xaxis.grid(True, color=C_LGREY, lw=0.35, alpha=0.6)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(fontsize=5.5, frameon=True, edgecolor=C_LGREY,
                  loc="lower right", handlelength=0.8)

    # ── (c) Radar / spider chart — model profiles across all 6 metrics ───
    _angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    _angles += _angles[:1]   # close the polygon

    ax_rad.set_theta_offset(np.pi / 2)
    ax_rad.set_theta_direction(-1)
    ax_rad.set_xticks(_angles[:-1])
    ax_rad.set_xticklabels(METRICS, fontsize=5.5, fontweight="bold",
                            color=C_DGREY)
    ax_rad.set_ylim(0, 1)
    ax_rad.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax_rad.set_yticklabels(["0.2", "0.4", "0.6", "0.8"],
                            fontsize=4.5, color=C_MGREY)
    ax_rad.grid(color=C_LGREY, linewidth=0.4, alpha=0.6)

    # Normalise each metric to [0,1] across models so radar is interpretable
    _metric_maxes = {m: max(results["overall"][l][m]["mean"]
                            for l in LABELS) or 1e-9
                     for m in METRICS}
    for lab in LABELS:
        vals = [results["overall"][lab][m]["mean"] / _metric_maxes[m]
                for m in METRICS]
        vals += vals[:1]
        col = _model_colors[lab]
        ax_rad.plot(_angles, vals, "o-", color=col, lw=1.0,
                    markersize=2.5, markeredgecolor=C_WHITE,
                    markeredgewidth=0.3)
        ax_rad.fill(_angles, vals, alpha=0.06, color=col)

    ax_rad.set_title("(c) Model Profile Radar\n(scores normalised per metric)",
                     fontsize=7, fontweight="bold", color=C_NAVY,
                     pad=14)

    fig_bs.suptitle(
        f"Zero-shot Retrieval: Bootstrap CI Visualisation  (N = {N:,}, "
        f"B = {CONFIG['bootstrap_B']})",
        fontsize=8, fontweight="bold", color=C_NAVY, y=1.01)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    _save(fig_bs, BENCH_DIR, "fig_bootstrap_ci")
    plt.show()

    # ── 4.13 LaTeX exports ───────────────────────────────────────────────
    print("[Export] LaTeX tables")

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Overall zero-shot retrieval performance on GeoQueryCA "
             f"(N = {N:,}). Values: mean (95\\% bootstrap CI, B={CONFIG['bootstrap_B']}). "
             r"Best per metric in bold.}",
             r"\label{tab:overall}", r"\small",
             r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
             "Model & " + " & ".join(METRICS) + r" \\", r"\midrule"]
    for i_lab, lab in enumerate(LABELS):
        row = [f"{lab} (Lexical)" if lab == "BM25" else lab]
        for j_m, m in enumerate(METRICS):
            d = results["overall"][lab][m]
            cell = f"{d['mean']:.4f}\\,({d['ci_lo']:.4f},{d['ci_hi']:.4f})"
            if best_row_for_col[j_m] - 1 == i_lab:
                cell = r"\textbf{" + cell + r"}"
            row.append(cell)
        lines.append(" & ".join(_tex_escape(x) for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(BENCH_DIR, "table3.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Cross-lingual retrieval by language group (in-pool).}",
             r"\label{tab:crosslingual}", r"\small",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Language Group & N & MRR & Recall@1 & Recall@5 \\",
             r"\midrule"]
    for lab in LABELS:
        for g in GROUPS_ORDER:
            gm = results["groups_inpool"][lab][g]
            if gm.get("N", 0) == 0:
                cells = [lab, g, "0", "—", "—", "—"]
            else:
                cells = [lab, g, f"{gm['N']:,}",
                         f"{gm['MRR']['mean']:.4f}",
                         f"{gm['Recall@1']['mean']:.4f}",
                         f"{gm['Recall@5']['mean']:.4f}"]
            if g == "Indian vernaculars":
                cells = [r"\textcolor{red!70!black}{" + _tex_escape(c) + "}"
                         for c in cells]
            else:
                cells = [_tex_escape(c) for c in cells]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if lab != LABELS[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(BENCH_DIR, "table4.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Pairwise MRR comparison. $\Delta = $ row $-$ column. "
             r"$p$-values from paired bootstrap ($B = " + str(CONFIG['bootstrap_B']) + "$"
             r"), Holm-Bonferroni corrected. *: $p<0.05$, **: $p<0.01$, ***: $p<0.001$.}",
             r"\label{tab:significance}", r"\small",
             r"\begin{tabular}{l" + "c" * n_m + "}", r"\toprule",
             "Model & " + " & ".join(LABELS) + r" \\", r"\midrule"]
    for i, A in enumerate(LABELS):
        cells = [A]
        for j, B_ in enumerate(LABELS):
            if i == j: cells.append("—")
            else:
                star = ""
                if p_mat[i, j] < 0.001: star = "***"
                elif p_mat[i, j] < 0.01: star = "**"
                elif p_mat[i, j] < 0.05: star = "*"
                cells.append(f"{delta_mat[i,j]:+.4f}{star}")
        lines.append(" & ".join(_tex_escape(c) for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(BENCH_DIR, "table5.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"  saved: table3.tex, table4.tex, table5.tex")

    # ── 4.14 Methodology MD ─────────────────────────────────────────────
    md = textwrap.dedent(f"""
    # GeoQueryCA Zero-shot Retrieval — Methodology

    **Generated:** {results['timestamp']}
    **Source CSV:** `{os.path.basename(CSV_PATH)}`
    **Evaluation set:** N = {N:,} (query, passage) pairs after deterministic
    deduplication on `(Contextual_Query, Cultural_Significance)` and
    null/whitespace filtering.

    ## Task formulation
    Given a multilingual contextual query, retrieve the corresponding monument's
    `Cultural_Significance` passage from N candidates. Exactly one passage is
    relevant per query (the row-matched ground truth), so:
    - **MRR** ≡ **MAP**
    - **Recall@k** ≡ **HitRate@k**
    - **nDCG@k** = E[1{{rank ≤ k}} / log₂(rank + 1)]  with IDCG = 1

    ## Models (see Table 2)
    | Label | Params | Embed. Dim | Retrieval-trained | Indian Lang Support | Source |
    |---|---|---|---|---|---|
    """)
    for mc in CONFIG["models"]:
        meta = MODEL_METADATA.get(mc["label"], {"params":"—","dim":"—",
                                                "retrieval_trained":"—",
                                                "indian_support":"—"})
        prefix_note = ""
        if mc.get("prefix_q") or mc.get("prefix_p"):
            prefix_note = (f" (prefixes: `{mc.get('prefix_q','')}` / "
                           f"`{mc.get('prefix_p','')}`)")
        md += (f"| {mc['label']} | {meta['params']} | {meta['dim']} | "
               f"{meta['retrieval_trained']} | {meta['indian_support']} | "
               f"`{mc['name']}`{prefix_note} |\n")
    md += textwrap.dedent(f"""

    All neural embeddings are L2-normalised, so the score matrix is cosine
    similarity. BM25 uses Okapi BM25 with unicode-safe `\\w+` tokenisation.

    ## Statistical methodology
    - **CIs:** percentile bootstrap, B={CONFIG['bootstrap_B']}, α={CONFIG['bootstrap_alpha']},
      seed={CONFIG['random_seed']}.
    - **Significance:** paired bootstrap (Berg-Kirkpatrick et al., 2012) on
      per-query MRR contributions, two-sided, H0-centred null.
    - **Multiple-comparison correction:** Holm-Bonferroni across all
      {len(sig_table)} pairwise comparisons.
    - **Reproducibility:** Python `random`, `numpy`, `torch` seeded with
      {CONFIG['random_seed']}; `cudnn.deterministic = True`.

    ## Cross-lingual breakdown
    1. **In-pool** (Table 4): rank within group only.
    2. **Global** (`metrics.json` → `groups_global`): rank against full N-pool.

    ## §3 Task formulation (for paper §3)

    GeoQueryCA is framed here as a **single-relevant-document retrieval task**:
    each contextual query in the dataset has exactly one ground-truth passage
    (the `Cultural_Significance` field of the same row). Under this
    simplification, several IR metrics collapse to equivalents — MRR is
    identical to Mean Average Precision (MAP), and Recall@k is identical to
    HitRate@k. This setup mirrors
    standard *passage retrieval* benchmarks such as Natural Questions and
    MS MARCO passage in their single-positive variants. We adopt it because:
    (i) each monument has a canonical cultural-significance description, so
    the one-to-one mapping is well-defined; (ii) it permits direct, fair
    comparison across the diverse model families considered here; and
    (iii) it produces stable, interpretable per-language metrics. We note
    that production retrieval systems would typically encounter multiple
    plausibly relevant passages per query; extending GeoQueryCA with
    graded relevance judgments is left to future work (see §6).

    ## §6 Limitations (for paper §6)

    1. **Single-relevant-doc assumption.** As detailed in §3, each query has
       a single canonical relevant passage. Real-world geographic retrieval
       may legitimately surface multiple related cultural descriptions
       (e.g., for a region containing several monuments). The metrics here
       therefore *upper-bound* failure rates rather than precisely
       characterise them.
    2. **Small per-language samples for under-represented languages.** Even
       after thresholding at n ≥ {CONFIG['min_language_n']}, several Indian
       vernaculars (Kannada, Malayalam, Marathi, Tamil) have n between 32
       and 50. Per-language MRR estimates on such small samples have wide
       confidence intervals and should be interpreted as exploratory rather
       than definitive.
    3. **No human relevance judgments.** Ground truth is derived from the
       dataset's row-level alignment between query and passage. We have not
       collected independent human relevance annotations to validate that
       the canonical passage is in fact the *most* relevant candidate, nor
       to identify near-duplicates among passages.
    4. **Zero-shot baselines reflect 2018–2024 model families.** Newer
       retrievers (multilingual-e5, BGE-M3) are included, but the
       comparison is not exhaustive. Cross-encoder rerankers, ColBERT-style
       late-interaction models, and instruction-tuned LLM retrievers are
       not evaluated here.
    5. **Cross-lingual setup is monolingual within each query.** All four
       language groups are evaluated with queries and passages in the same
       language. True cross-lingual retrieval (e.g., Tamil query against
       English passages) is supported by the data but not the principal
       evaluation here.
    """)
    with open(os.path.join(BENCH_DIR, "METHODOLOGY.md"), "w") as f:
        f.write(md.lstrip())

    benchmark_summary = {"LABELS": LABELS, "METRICS": METRICS,
                         "overall": results["overall"],
                         "sig_table": sig_table,
                         "geo_error": results.get("geo_error", {})}
    print(f"\nPhase 2 outputs → {BENCH_DIR}/")

# ════════════════════════════════════════════════════════════════════════════
#         PHASE 3 — FINE-TUNING EXPERIMENT  (mBERT → Sentence-BERT-style)
# ════════════════════════════════════════════════════════════════════════════
#  Adds one row to the headline benchmark by fine-tuning an mBERT-class
#  encoder with MultipleNegativesRankingLoss on a landmark-grouped train/test
#  split (GroupShuffleSplit, 30% of landmarks to test, seed 42), then
#  evaluating on the test set. The same test set is re-evaluated
#  zero-shot for *all* models in CONFIG["models"] so the comparison is
#  apples-to-apples (same N_test). Produces:
#      table6_finetune_vs_zeroshot.{png,pdf}
#      fig_finetune_improvement.{png,pdf}
#      table6.tex
# ════════════════════════════════════════════════════════════════════════════
finetune_summary = None
print("\n" + "═"*72 + "\nPHASE 3 — Fine-tuning Experiment\n" + "═"*72)
if not CONFIG.get("run_finetuning"):
    print("Skipped: CONFIG['run_finetuning'] is False. "
          "Set it to True in CONFIG and re-run.")
elif not CONFIG.get("run_benchmark"):
    print("Skipped: requires CONFIG['run_benchmark']=True "
          "(Phase 3 uses helpers defined inside Phase 2).")
else:
    print(f"Base model: {CONFIG['finetune_base_model']}")
    print(f"Epochs: {CONFIG['finetune_epochs']}   "
          f"Batch size: {CONFIG['finetune_batch_size']}   "
          f"Test size: {CONFIG['finetune_test_size']:.0%}")
    print("Starting Phase 3…")

if CONFIG.get("run_finetuning") and CONFIG["run_benchmark"]:
    from sklearn.model_selection import train_test_split
    from sentence_transformers import (SentenceTransformer, InputExample,
                                       losses, models as st_models)
    from torch.utils.data import DataLoader

    # Defensive defaults — any missing fine-tuning CONFIG key gets a sane value,
    # so users pasting CONFIG into Colab cells can't hit KeyError mid-Phase-3.
    _FT_DEFAULTS = {
        "finetune_base_model":  "bert-base-multilingual-cased",
        "finetune_epochs":      3,
        "finetune_batch_size":  16,
        "finetune_warmup_steps":100,
        "finetune_test_size":   0.30,
    }
    for _k, _v in _FT_DEFAULTS.items():
        if _k not in CONFIG:
            print(f"  CONFIG['{_k}'] missing → using default {_v!r}")
            CONFIG[_k] = _v

    # ── 5.1 Leak-free split grouped by Monument ──────────────────────────
    # GeoQueryCA deliberately reuses the same Cultural_Significance passage
    # across multiple queries (the "hard negatives" contribution). A plain
    # row-wise split therefore lets a passage that was a training TARGET
    # reappear as a test-time retrieval target — the model has already seen
    # the exact gold text, which inflates fine-tuned test metrics.
    #
    # Fix: split by the grouping key (Monument when available, else the
    # passage text itself), so an entire monument — and thus its passage —
    # lands wholly in train OR wholly in test, never both. Rare-language rows
    # are still forced into train. A final guard drops any test row whose
    # passage still appears in train, guaranteeing zero passage overlap.
    from sklearn.model_selection import GroupShuffleSplit

    group_key = "Monument" if "Monument" in df.columns else "Cultural_Significance"
    _df = df.reset_index(drop=True).copy()
    _df["_grp"] = _df[group_key].astype("string").fillna(
        _df["Cultural_Significance"].astype("string"))

    # Force rare-language rows (and their whole group) into train.
    lang_counts = _df["Query_Language"].value_counts()
    rare_langs  = lang_counts[lang_counts < 2].index.tolist()
    rare_groups = set(_df.loc[_df["Query_Language"].isin(rare_langs), "_grp"])
    forced_train = _df["_grp"].isin(rare_groups)
    df_forced = _df[forced_train]
    df_splitable = _df[~forced_train]

    gss = GroupShuffleSplit(n_splits=1,
                            test_size=CONFIG["finetune_test_size"],
                            random_state=CONFIG["random_seed"])
    tr_idx, te_idx = next(gss.split(df_splitable,
                                    groups=df_splitable["_grp"]))
    train_df = pd.concat([df_splitable.iloc[tr_idx], df_forced],
                         ignore_index=True)
    test_df  = df_splitable.iloc[te_idx].reset_index(drop=True)

    # Guard: remove any test row whose passage leaked into train.
    _train_passages = set(train_df["Cultural_Significance"].astype("string"))
    _leak_mask = test_df["Cultural_Significance"].astype("string").isin(_train_passages)
    n_leak = int(_leak_mask.sum())
    if n_leak:
        print(f"  Dropped {n_leak} test row(s) whose passage also appears in train.")
        test_df = test_df[~_leak_mask].reset_index(drop=True)

    train_df = train_df.drop(columns=["_grp"], errors="ignore")
    test_df  = test_df.drop(columns=["_grp"], errors="ignore")
    N_train, N_test = len(train_df), len(test_df)

    # Sanity check: passage sets must be disjoint.
    _overlap = (set(train_df["Cultural_Significance"].astype("string"))
                & set(test_df["Cultural_Significance"].astype("string")))
    assert not _overlap, f"Passage leak remains: {len(_overlap)} shared passages"

    print(f"Train: {N_train:,}   Test: {N_test:,}   "
          f"(grouped by '{group_key}', rare langs forced to train: "
          f"{int(forced_train.sum())} rows)   passage overlap: 0 ✓")

    test_queries  = test_df["Contextual_Query"].tolist()
    test_passages = test_df["Cultural_Significance"].tolist()
    test_langs    = test_df["Query_Language"].to_numpy()
    test_groups   = test_df["group"].to_numpy()

    # ── 5.2 Fine-tune mBERT-init Sentence-BERT ───────────────────────────
    base = CONFIG["finetune_base_model"]
    ft_label = f"FT-{base.split('/')[-1].split('-')[0]}".replace("FT-bert", "FT-mBERT")
    print(f"\nFine-tuning {base} → {ft_label}")
    ft_path = os.path.join(BENCH_DIR, f"{ft_label.lower()}-finetuned")
    if os.path.exists(ft_path):
        print(f"  [cache] re-using fine-tuned model at {ft_path}")
        ft_model = SentenceTransformer(ft_path, device=str(DEVICE))
    else:
        # Build a SentenceTransformer wrapping the base model
        word = st_models.Transformer(base, max_seq_length=CONFIG["max_seq_length"])
        pool = st_models.Pooling(word.get_word_embedding_dimension(),
                                 pooling_mode_mean_tokens=True)
        ft_model = SentenceTransformer(modules=[word, pool], device=str(DEVICE))

        train_examples = [
            InputExample(texts=[row["Contextual_Query"],
                                row["Cultural_Significance"]])
            for _, row in train_df.iterrows()
        ]
        loader  = DataLoader(train_examples,
                             batch_size=CONFIG["finetune_batch_size"], shuffle=True)
        loss_fn = losses.MultipleNegativesRankingLoss(ft_model)
        ft_model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=CONFIG["finetune_epochs"],
            warmup_steps=CONFIG["finetune_warmup_steps"],
            show_progress_bar=True,
        )
        ft_model.save(ft_path)
        print(f"  saved fine-tuned model → {ft_path}")

    # ── 5.3 Encode test set with fine-tuned model + score ─────────────────
    q_ft = ft_model.encode(test_queries,  batch_size=CONFIG["batch_size"],
                           convert_to_tensor=True, show_progress_bar=True,
                           normalize_embeddings=True).to(DEVICE)
    p_ft = ft_model.encode(test_passages, batch_size=CONFIG["batch_size"],
                           convert_to_tensor=True, show_progress_bar=True,
                           normalize_embeddings=True).to(DEVICE)
    S_ft = q_ft @ p_ft.T
    r_ft = ranks_from_scores(S_ft)
    pq_ft = per_query_metric_arrays(r_ft)

    # ── Geo-error for fine-tuned model (test split) ───────────────────────
    # test_df was built with reset_index(drop=True), so test_df.index is
    # 0..N_test-1 and does NOT map back to df rows. We therefore read the
    # coordinates straight off test_df, which carries Lat_DD/Lon_DD as columns
    # (attached at split time), guaranteeing per-row alignment with r_ft.
    ft_geo_results = {}
    if HAS_COORDS:
        if {"Lat_DD", "Lon_DD"}.issubset(test_df.columns):
            test_lat = test_df["Lat_DD"].to_numpy(dtype=np.float64)
            test_lon = test_df["Lon_DD"].to_numpy(dtype=np.float64)
        else:
            # Fallback: reindex by matching the (query, passage) key.
            merged = test_df[["Contextual_Query", "Cultural_Significance"]].merge(
                df[["Contextual_Query", "Cultural_Significance", "Lat_DD", "Lon_DD"]],
                on=["Contextual_Query", "Cultural_Significance"], how="left")
            test_lat = merged["Lat_DD"].to_numpy(dtype=np.float64)
            test_lon = merged["Lon_DD"].to_numpy(dtype=np.float64)
        try:
            S_ft_np = S_ft.cpu().numpy().astype(np.float32)
            ft_geo_results = geo_error_metrics_from_score_matrix(
                S_ft_np, test_lat, test_lon, r_ft)
            print(f"  FT Geo: MAE={ft_geo_results['MAE_geo_km']:.1f} km  "
                  f"GeoAcc@161km={ft_geo_results.get('GeoAcc@161km',0)*100:.1f}%  "
                  f"geo-wMRR={ft_geo_results.get('geo_weighted_MRR',float('nan')):.4f}")
        except Exception as _ge:
            print(f"  ⚠  Geo-error for fine-tuned model failed: {_ge}")

    del ft_model, q_ft, p_ft, S_ft
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # ── 5.4 Re-evaluate every zero-shot model on the SAME test split ─────
    # We can reuse the per_query_metrics arrays (computed on full N) by
    # indexing them with the test_df's original row positions. To get those
    # positions, map test_df rows back to the cleaned df index.
    df_keyed = df.reset_index().rename(columns={"index": "_orig"})
    test_keys = test_df.merge(
        df_keyed[["Contextual_Query","Cultural_Significance","_orig"]],
        on=["Contextual_Query","Cultural_Significance"], how="left")["_orig"].to_numpy()

    # NOTE: per-query metrics were computed against the FULL N-pool.
    # For a fair comparison we re-compute zero-shot metrics on the test
    # pool only (test queries × test passages), so all models are ranking
    # over the same N_test candidates.
    print("\nRe-evaluating zero-shot models on test-pool only…")
    zs_test_metrics = {}
    # Only re-evaluate models that succeeded in Phase 2.
    phase3_models = [mc for mc in CONFIG["models"] if mc["label"] in LABELS]
    for mc in phase3_models:
        lab = mc["label"]
        try:
            # Build per-model score matrix on TEST subset by re-encoding.
            if mc["kind"] == "bm25":
                from rank_bm25 import BM25Okapi
                tok = lambda t: re.findall(r"\w+", t.lower(), flags=re.UNICODE)
                bm = BM25Okapi([tok(p) for p in test_passages])
                S = np.zeros((N_test, N_test), dtype=np.float32)
                for i in range(N_test):
                    S[i] = bm.get_scores(tok(test_queries[i]))
                S_t = torch.from_numpy(S).to(DEVICE)
            else:
                prefix_q = mc.get("prefix_q", ""); prefix_p = mc.get("prefix_p", "")
                if mc["kind"] == "hf":
                    from transformers import AutoModel, AutoTokenizer
                    tok_t = AutoTokenizer.from_pretrained(mc["name"])
                    mdl_t = AutoModel.from_pretrained(mc["name"]).to(DEVICE).eval()
                    def _enc(txts, pf):
                        if pf: txts = [pf + t for t in txts]
                        out_emb = []
                        with torch.no_grad():
                            for i in range(0, len(txts), CONFIG["batch_size"]):
                                enc = tok_t(txts[i:i+CONFIG["batch_size"]],
                                            padding=True, truncation=True,
                                            max_length=CONFIG["max_seq_length"],
                                            return_tensors="pt").to(DEVICE)
                                o = mdl_t(**enc)
                                tok_emb = o[0]
                                m = enc["attention_mask"].unsqueeze(-1).expand(
                                        tok_emb.size()).float()
                                pooled = (tok_emb * m).sum(1) / m.sum(1).clamp(min=1e-9)
                                out_emb.append(F.normalize(pooled, p=2, dim=1))
                        return torch.cat(out_emb, dim=0)
                    q_e = _enc(test_queries,  prefix_q)
                    p_e = _enc(test_passages, prefix_p)
                    del mdl_t, tok_t
                else:
                    from sentence_transformers import SentenceTransformer
                    m_t = SentenceTransformer(mc["name"], device=str(DEVICE),
                                              trust_remote_code=True)
                    q_e = m_t.encode([prefix_q+t for t in test_queries]
                                     if prefix_q else test_queries,
                                     batch_size=CONFIG["batch_size"],
                                     convert_to_tensor=True,
                                     show_progress_bar=False,
                                     normalize_embeddings=True).to(DEVICE)
                    p_e = m_t.encode([prefix_p+t for t in test_passages]
                                     if prefix_p else test_passages,
                                     batch_size=CONFIG["batch_size"],
                                     convert_to_tensor=True,
                                     show_progress_bar=False,
                                     normalize_embeddings=True).to(DEVICE)
                    del m_t
                S_t = q_e @ p_e.T
                del q_e, p_e
            r_t = ranks_from_scores(S_t)
            zs_test_metrics[lab] = per_query_metric_arrays(r_t)
            del S_t
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            print(f"  ✓ {lab} test-pool: "
                  f"MRR={zs_test_metrics[lab]['MRR'].mean():.4f}")
        except Exception as e:
            print(f"  ⚠  {lab} test-pool re-eval failed "
                  f"({type(e).__name__}); skipping in Phase 3 comparison.")
            if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # ── 5.5 Aggregate test-pool comparison with CIs ──────────────────────
    ft_results = {}
    for lab, pq in zs_test_metrics.items():
        ft_results[lab + " (zero-shot)"] = metrics_with_ci(pq)
    ft_results[ft_label + " (fine-tuned)"] = metrics_with_ci(pq_ft)

    # Improvement deltas vs the best zero-shot baseline on this test pool
    if zs_test_metrics:
        best_zs_label = max(zs_test_metrics,
                            key=lambda l: zs_test_metrics[l]["MRR"].mean())
        delta_mrr = (pq_ft["MRR"].mean() -
                     zs_test_metrics[best_zs_label]["MRR"].mean())
    else:
        best_zs_label = "—"
        delta_mrr = float("nan")

    # Also per-group MRR for the fine-tuned model
    ft_group_metrics = {}
    for g in GROUPS_ORDER:
        idx = np.where(test_groups == g)[0]
        if len(idx) == 0:
            ft_group_metrics[g] = {"N": 0}; continue
        pq_g = {k: v[idx] for k, v in pq_ft.items()}
        ft_group_metrics[g] = {"N": int(len(idx)), **metrics_with_ci(pq_g)}

    # ── 5.6 Save fine-tuning metrics JSON ────────────────────────────────
    ft_json = {"base_model": base,
               "epochs": CONFIG["finetune_epochs"],
               "batch_size": CONFIG["finetune_batch_size"],
               "N_train": N_train, "N_test": N_test,
               "results": ft_results,
               "ft_group_metrics": ft_group_metrics,
               "best_zero_shot_on_test": best_zs_label,
               "delta_MRR_vs_best_zs": float(delta_mrr),
               "geo_error_finetuned": ft_geo_results if HAS_COORDS else {}}
    with open(os.path.join(BENCH_DIR, "finetune_metrics.json"), "w") as f:
        json.dump(_jsonsafe(ft_json), f, indent=2)

    # ── 5.7 Table 6 — fine-tune vs zero-shot on common test split ────────
    print("\n[Render] Table 6 (fine-tune vs zero-shot, test split)")
    t6_cols = ["Model"] + METRICS
    t6_data = []
    rows_order = [l + " (zero-shot)" for l in LABELS if l in zs_test_metrics] \
                 + [ft_label + " (fine-tuned)"]
    for lab in rows_order:
        o = ft_results[lab]
        t6_data.append([lab] + [_fmt_ci(o[m]) for m in METRICS])
    means6 = np.array([[ft_results[r][m]["mean"] for m in METRICS]
                       for r in rows_order])
    best6  = means6.argmax(axis=0) + 1

    fig, ax = plt.subplots(figsize=(SN_2COL, 0.20 * len(rows_order) + 0.9))
    fig.patch.set_facecolor(WHITE); ax.axis("off")
    table = ax.table(cellText=t6_data, colLabels=t6_cols, cellLoc="center",
                     loc="center",
                     colWidths=[0.22] + [0.13] * len(METRICS))
    table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.8)
    for j in range(len(t6_cols)):
        c = table[0, j]; c.set_facecolor(C_NAVY)
        c.set_text_props(color=TEXT_LIGHT, fontweight="bold")
        c.set_edgecolor(WHITE); c.set_linewidth(0.5)
    for i in range(1, len(t6_data) + 1):
        is_ft = "fine-tuned" in t6_data[i-1][0]
        bg = C_LGRN if is_ft else (C_VBLUE if i % 2 else BLUE_ALT)
        for j in range(len(t6_cols)):
            c = table[i, j]; c.set_facecolor(bg)
            c.set_edgecolor(WHITE); c.set_linewidth(0.4)
            if j == 0:
                c.set_text_props(fontweight="bold",
                                 color=C_BLUE2 if is_ft else C_NAVY)
            elif best6[j-1] == i:
                c.set_facecolor(HIGHLIGHT)
                c.set_text_props(fontweight="bold", color=C_NAVY)
            else:
                c.set_text_props(color=TEXT_DARK)
    ax.set_title(f"Table 6   Fine-tuning vs zero-shot on shared test split "
                 f"(N_test = {N_test:,})",
                 fontsize=8, fontweight="bold", color=C_NAVY, pad=6, loc="left")
    ax.text(0, 0.99,
            f"All rows ranked over the same test pool of {N_test} candidates. "
            f"Fine-tuned model trained on {N_train:,} pairs, "
            f"{CONFIG['finetune_epochs']} epochs. Green row = fine-tuned.",
            transform=ax.transAxes, fontsize=5.5, color=C_MGREY, style="italic")
    plt.tight_layout(); _save(fig, BENCH_DIR, "table6_finetune_vs_zeroshot"); plt.show()

    # ════════════════════════════════════════════════════════════════════════
    # ── NEW: Table 3b — Overall benchmark in reference-image style ─────────
    # Columns: Model | MRR | nDCG@10 | Recall@1 | Recall@5 | Recall@10
    # Rows:    one per zero-shot model (BM25, mBERT, XLM-R, LaBSE, …)
    # Best value per column bold + highlighted (matching reference image)
    # ════════════════════════════════════════════════════════════════════════
    print("\n[Render] Table 3b — Overall benchmark (reference-image style)")

    # Metric columns matching the reference image exactly
    # Recall@k = top-k hit rate
    _T3B_METRICS = [
        ("MRR",        "MRR"),
        ("nDCG@10",    "nDCG@10"),
        ("Recall@1",   "Recall@1"),
        ("Recall@5",   "Recall@5"),
        ("Recall@10",  "Recall@10"),
    ]
    _T3B_DISPLAY_COLS = ["Model"] + [d for d, _ in _T3B_METRICS]
    _T3B_INTERNAL     = [k for _, k in _T3B_METRICS]

    # Build data rows
    _t3b_data  = []
    _t3b_means = []
    for lab in LABELS:
        o = results["overall"][lab]
        disp = f"{lab} (Lexical)" if lab == "BM25" else lab
        row_vals = [o[m]["mean"] for m in _T3B_INTERNAL]
        _t3b_means.append(row_vals)
        _t3b_data.append(
            [disp] + [f"{v:.4f}" for v in row_vals])

    _t3b_arr  = np.array(_t3b_means)
    _t3b_best = _t3b_arr.argmax(axis=0) + 1   # 1-based (row 0 = header)

    # ── Draw table ─────────────────────────────────────────────────────────
    _col_w3b = [0.22] + [0.156] * len(_T3B_METRICS)
    fig, ax = plt.subplots(figsize=(SN_2COL, 0.22 * len(LABELS) + 1.1))
    fig.patch.set_facecolor(C_WHITE); ax.axis("off")

    table3b = ax.table(cellText=_t3b_data, colLabels=_T3B_DISPLAY_COLS,
                       cellLoc="center", loc="center",
                       colWidths=_col_w3b)
    table3b.auto_set_font_size(False); table3b.set_fontsize(6.5)
    table3b.scale(1, 2.0)

    # Header row
    for j in range(len(_T3B_DISPLAY_COLS)):
        c = table3b[0, j]
        c.set_facecolor(C_NAVY); c.set_edgecolor(C_WHITE); c.set_linewidth(0.6)
        c.set_text_props(color=TEXT_LIGHT, fontweight="bold")

    # Data rows
    for i in range(1, len(_t3b_data) + 1):
        bg = C_VBLUE if i % 2 else BLUE_ALT
        for j in range(len(_T3B_DISPLAY_COLS)):
            c = table3b[i, j]
            c.set_edgecolor(C_WHITE); c.set_linewidth(0.4)
            if j == 0:
                c.set_facecolor(bg)
                c.set_text_props(fontweight="bold", color=C_NAVY)
            elif _t3b_best[j - 1] == i:
                # Bold + highlighted — matching reference image
                c.set_facecolor(HIGHLIGHT)
                c.set_text_props(fontweight="bold", color=C_NAVY)
            else:
                c.set_facecolor(bg)
                c.set_text_props(color=TEXT_DARK)

    ax.set_title(
        f"Table 3: Overall benchmark results on GeoQueryCA (N = {N:,}). "
        "All models evaluated zero-shot.",
        fontsize=7, fontweight="bold", color=C_NAVY, pad=6, loc="left",
        wrap=True)
    ax.text(0, 0.96,
            "Recall@k (fraction of queries with correct doc in top-k). "
            "Bold = best per column.",
            transform=ax.transAxes, fontsize=5.5, color=C_MGREY, style="italic")

    plt.tight_layout()
    _save(fig, BENCH_DIR, "table3b_overall_benchmark_refstyle")
    plt.show()
    # ────────────────────────────────────────────────────────────────────────

    # ════════════════════════════════════════════════════════════════════════
    # ── NEW: Table 5b — Zero-shot vs fine-tuned in reference-image style ───
    # Columns: Metric | Zero-shot | Fine-tuned | Improvement (Δ)
    # Rows:    MRR, Recall@1, Recall@5, Recall@10
    # Matches reference image Table 5 exactly
    # ════════════════════════════════════════════════════════════════════════
    print("[Render] Table 5b — Zero-shot vs fine-tuned (reference-image style)")

    _T5B_METRICS = [
        ("MRR",       "MRR"),
        ("Recall@1",  "Recall@1"),
        ("Recall@5",  "Recall@5"),
        ("Recall@10", "Recall@10"),
    ]

    # Best zero-shot result per metric (across all baselines on the TEST pool)
    _t5b_data = []
    for disp_name, internal_key in _T5B_METRICS:
        if internal_key not in ft_results.get(ft_label + " (fine-tuned)", {}):
            continue
        # Zero-shot: take the best baseline on the test split
        _zs_vals = [ft_results[l + " (zero-shot)"][internal_key]["mean"]
                    for l in LABELS if l + " (zero-shot)" in ft_results
                    and internal_key in ft_results[l + " (zero-shot)"]]
        if not _zs_vals:
            continue
        zs_best = max(_zs_vals)
        ft_val  = ft_results[ft_label + " (fine-tuned)"][internal_key]["mean"]
        delta   = ft_val - zs_best
        _t5b_data.append([
            disp_name,
            f"{zs_best:.4f}",
            f"{ft_val:.4f}",
            f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}",
        ])

    if _t5b_data:
        _T5B_COLS = ["Metric", "Zero-shot", "Fine-tuned", "Improvement"]
        _col_w5b  = [0.22, 0.22, 0.22, 0.22]

        fig, ax = plt.subplots(figsize=(SN_15COL, 0.22 * len(_t5b_data) + 1.1))
        fig.patch.set_facecolor(C_WHITE); ax.axis("off")

        table5b = ax.table(cellText=_t5b_data, colLabels=_T5B_COLS,
                           cellLoc="center", loc="center",
                           colWidths=_col_w5b)
        table5b.auto_set_font_size(False); table5b.set_fontsize(6.5)
        table5b.scale(1, 2.0)

        # Header
        for j in range(len(_T5B_COLS)):
            c = table5b[0, j]
            c.set_facecolor(C_NAVY); c.set_edgecolor(C_WHITE); c.set_linewidth(0.6)
            c.set_text_props(color=TEXT_LIGHT, fontweight="bold")

        # Data rows — Fine-tuned column and Improvement column bolded
        for i in range(1, len(_t5b_data) + 1):
            bg = C_VBLUE if i % 2 else BLUE_ALT
            for j in range(len(_T5B_COLS)):
                c = table5b[i, j]
                c.set_edgecolor(C_WHITE); c.set_linewidth(0.4)
                if j == 0:
                    # Metric name
                    c.set_facecolor(bg)
                    c.set_text_props(fontweight="bold", color=C_NAVY)
                elif j == 1:
                    # Zero-shot column — normal weight
                    c.set_facecolor(bg)
                    c.set_text_props(color=TEXT_DARK)
                elif j == 2:
                    # Fine-tuned column — bold (best value, matching reference)
                    c.set_facecolor(HIGHLIGHT)
                    c.set_text_props(fontweight="bold", color=C_NAVY)
                elif j == 3:
                    # Improvement (Δ) — bold, positive = darker blue
                    c.set_facecolor(C_PVBLUE)
                    c.set_text_props(fontweight="bold", color=C_NAVY)

        ax.set_title(
            f"Table 5: Zero-shot vs. fine-tuned {ft_label.replace('FT-','')} "
            f"on GeoQueryCA test set (N = {N_test:,}).",
            fontsize=7, fontweight="bold", color=C_NAVY, pad=6, loc="left")
        ax.text(0, 0.96,
                f"Zero-shot = best baseline ({best_zs_label}) on the same test pool. "
                "Fine-tuned values bold. Δ = fine-tuned − zero-shot.",
                transform=ax.transAxes, fontsize=5.5, color=C_MGREY, style="italic")

        plt.tight_layout()
        _save(fig, BENCH_DIR, "table5b_zeroshot_vs_finetuned_refstyle")
        plt.show()
    else:
        print("  [Table 5b] skipped — fine-tuning metrics not available")
    # ────────────────────────────────────────────────────────────────────────

    # ════════════════════════════════════════════════════════════════════════
    # ── NEW: Table 6b — Cross-lingual breakdown after fine-tuning ──────────
    # Matches reference image Table 6 exactly:
    # Columns: Language Group | N | MRR | Recall@1 | Recall@5
    # Rows:    English, Indian vernaculars (red), European, Others
    # Indian vernaculars row coloured red (matching reference image)
    # ════════════════════════════════════════════════════════════════════════
    print("[Render] Table 6b — Cross-lingual breakdown after fine-tuning")

    _T6B_COLS    = ["Language Group", "N", "MRR", "Recall@1", "Recall@5"]
    _T6B_METRICS = ["MRR", "Recall@1", "Recall@5"]

    _t6b_data   = []
    _t6b_means  = []
    _t6b_styles = []

    for g in GROUPS_ORDER:
        gm = ft_group_metrics.get(g, {})
        n  = gm.get("N", 0)
        if n == 0:
            continue
        vals = []
        for mk in _T6B_METRICS:
            v = gm.get(mk, {})
            vals.append(v.get("mean", float("nan"))
                        if isinstance(v, dict) else float("nan"))
        _t6b_means.append(vals)
        _t6b_data.append([
            g, str(n),
            f"{vals[0]:.4f}", f"{vals[1]:.4f}", f"{vals[2]:.4f}",
        ])
        _t6b_styles.append("indian" if g == "Indian vernaculars" else "normal")

    if _t6b_data:
        _t6b_arr  = np.array(_t6b_means)
        _t6b_best = _t6b_arr.argmax(axis=0) + 1   # 1-based, numeric cols only

        _C_INDIAN_BG = "#FCE4D6"
        _C_INDIAN_FG = "#843C0C"
        _col_w6b = [0.28, 0.10, 0.20, 0.20, 0.20]

        fig, ax = plt.subplots(
            figsize=(SN_15COL, 0.22 * len(_t6b_data) + 1.1))
        fig.patch.set_facecolor(C_WHITE); ax.axis("off")

        table6b = ax.table(cellText=_t6b_data, colLabels=_T6B_COLS,
                           cellLoc="center", loc="center",
                           colWidths=_col_w6b)
        table6b.auto_set_font_size(False); table6b.set_fontsize(6.5)
        table6b.scale(1, 2.0)

        # Header
        for j in range(len(_T6B_COLS)):
            c = table6b[0, j]
            c.set_facecolor(C_NAVY); c.set_edgecolor(C_WHITE); c.set_linewidth(0.6)
            c.set_text_props(color=TEXT_LIGHT, fontweight="bold")

        # Data rows
        for i, (row, style) in enumerate(zip(_t6b_data, _t6b_styles), start=1):
            is_indian = (style == "indian")
            bg_row = _C_INDIAN_BG if is_indian else (C_VBLUE if i % 2 else BLUE_ALT)
            fg_row = _C_INDIAN_FG if is_indian else TEXT_DARK

            for j in range(len(_T6B_COLS)):
                c = table6b[i, j]
                c.set_edgecolor(C_WHITE); c.set_linewidth(0.4)
                if j <= 1:
                    c.set_facecolor(bg_row)
                    c.set_text_props(
                        fontweight="bold" if j == 0 else "normal",
                        color=fg_row)
                else:
                    mi = j - 2   # metric index
                    is_best = (_t6b_best[mi] == i)
                    c.set_facecolor(HIGHLIGHT if is_best else bg_row)
                    c.set_text_props(
                        fontweight="bold" if (is_best or is_indian) else "normal",
                        color=(_C_INDIAN_FG if is_indian else C_NAVY)
                              if is_best else fg_row)

        ax.set_title(
            f"Table 6: Cross-lingual retrieval breakdown after fine-tuning "
            f"{ft_label.replace('FT-','')} on GeoQueryCA (N\u2009=\u2009{N_test:,}).",
            fontsize=7, fontweight="bold", color=C_NAVY, pad=6, loc="left")
        ax.text(0, 0.96,
                "Recall@k = top-k hit rate.  "
                "Red rows: Indian vernaculars.  Bold = best per column.",
                transform=ax.transAxes, fontsize=5.5,
                color=C_MGREY, style="italic")

        plt.tight_layout()
        _save(fig, BENCH_DIR, "table6b_crosslingual_finetuned")
        plt.show()
    else:
        print("  [Table 6b] skipped — ft_group_metrics not available")
    # ────────────────────────────────────────────────────────────────────────
    ft_grp_mrr = []
    zs_grp_mrr = []
    grp_labels = []
    for g in GROUPS_ORDER:
        idx = np.where(test_groups == g)[0]
        if len(idx) == 0: continue
        grp_labels.append(g)
        ft_grp_mrr.append(float(pq_ft["MRR"][idx].mean()))
        zs_grp_mrr.append(float(zs_test_metrics[best_zs_label]["MRR"][idx].mean()))

    x = np.arange(len(grp_labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(SN_2COL, SN_2COL * 0.40))
    fig.patch.set_facecolor(C_WHITE); ax.set_facecolor(C_WHITE)
    b1 = ax.bar(x - w/2, zs_grp_mrr, width=w, color=C_NAVY,
                label=f"Best zero-shot ({best_zs_label})",
                edgecolor=WHITE, linewidth=0.5, zorder=3)
    b2 = ax.bar(x + w/2, ft_grp_mrr, width=w, color=C_LBLUE,
                label=f"{ft_label} (fine-tuned)",
                edgecolor=WHITE, linewidth=0.5, zorder=3)
    ymax = max(zs_grp_mrr + ft_grp_mrr) if (zs_grp_mrr + ft_grp_mrr) else 0.1
    ax.set_ylim(0, ymax * 1.28)
    for bar, v in zip(b1, zs_grp_mrr):
        ax.text(bar.get_x() + bar.get_width()/2, v + ymax*0.04,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=6, color=C_NAVY, fontweight="bold")
    for bar, v in zip(b2, ft_grp_mrr):
        ax.text(bar.get_x() + bar.get_width()/2, v + ymax*0.04,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=6, color=C_LBLUE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(grp_labels, fontsize=7,
                                         fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel("MRR (test pool)")
    ax.set_title(f"Fine-tuning effect by language group "
                 f"(test pool, N_test = {N_test:,})",
                 fontsize=8, fontweight="bold", color=C_NAVY, pad=6)
    ax.legend(fontsize=6, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); _save(fig, BENCH_DIR, "fig_finetune_improvement"); plt.show()

    # ── 5.9 LaTeX export for Table 6 ─────────────────────────────────────
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Fine-tuning effect: fine-tuned " + ft_label.replace("_","-") +
             r" vs all zero-shot baselines on a shared 30\% test split (N\_{test} = "
             + str(N_test) + r"). Best per metric in bold; fine-tuned row "
             r"shaded in the figure.}",
             r"\label{tab:finetune}", r"\small",
             r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
             "Model & " + " & ".join(METRICS) + r" \\", r"\midrule"]
    for i_lab, lab in enumerate(rows_order):
        cells = [lab.replace("_","-")]
        for j_m, m in enumerate(METRICS):
            d = ft_results[lab][m]
            cell = f"{d['mean']:.4f}\\,({d['ci_lo']:.4f},{d['ci_hi']:.4f})"
            if best6[j_m] - 1 == i_lab:
                cell = r"\textbf{" + cell + r"}"
            cells.append(cell)
        if "fine-tuned" in lab:
            lines.append(r"\midrule")
        lines.append(" & ".join(_tex_escape(c) for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(BENCH_DIR, "table6.tex"), "w") as f:
        f.write("\n".join(lines))

    print(f"\nFine-tuning Δ MRR vs best zero-shot ({best_zs_label}): "
          f"{delta_mrr:+.4f}")
    finetune_summary = {"ft_label": ft_label, "N_train": N_train,
                        "N_test": N_test, "delta_MRR": float(delta_mrr),
                        "best_zs_label": best_zs_label,
                        "ft_group_metrics": ft_group_metrics,
                        "geo_error_finetuned": ft_geo_results}

    # Re-render Table 2 with the fine-tuned model appended.
    # The original (zero-shot only) figure stays as table2_model_characteristics.png;
    # this version is the "full Methods table" the paper should cite.
    print("\n[Render] Table 2 (model characteristics + fine-tuned)")
    render_table2(LABELS + [ft_label], BENCH_DIR,
                  basename="table2_model_characteristics_with_finetuned",
                  title_suffix="   (incl. fine-tuned model)")

    print(f"Phase 3 outputs → {BENCH_DIR}/  "
          f"(table6, finetune_metrics.json, fig_finetune_improvement, "
          f"table2_model_characteristics_with_finetuned)")

# ════════════════════════════════════════════════════════════════════════════
# 5. Unified summary report
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72 + "\nWriting unified summary report\n" + "═"*72)
report = [f"# GeoQueryCA — End-to-End Report",
          f"\n**Source:** `{os.path.basename(CSV_PATH)}`  ",
          f"**Records:** {N:,}  ",
          f"**Generated:** {datetime.utcnow().isoformat()}\n"]
if profiling_summary is not None:
    q = profiling_summary["metrics"]["quality"]
    div = profiling_summary["metrics"]["diversity"]
    co = profiling_summary["metrics"].get("coordinates", {})
    report += ["## Phase 1 — Dataset Profiling",
               f"- Completeness: **{q['completeness_overall_pct']:.2f}%**, "
               f"duplicate rows: **{q['duplicate_rows']}**",
               f"- Unique countries: **{div['Country']['unique']}**, "
               f"languages: **{div['Query_Language']['unique']}**, "
               f"categories: **{div['Category']['unique']}**"]
    if co.get("has_coordinates"):
        report.append(f"- Coordinates: **{co['lat_dd_valid_n']:,}/{N:,}** valid "
                      f"({co['lat_dd_valid_pct']:.1f}%) | "
                      f"Lat [{co['lat_range'][0]:.2f}°, {co['lat_range'][1]:.2f}°]  "
                      f"Lon [{co['lon_range'][0]:.2f}°, {co['lon_range'][1]:.2f}°]")
    report.append(f"- Outputs: `{DESC_DIR}/` (8 figures + JSON + MD)\n")

if coord_validation_summary is not None:
    cv = coord_validation_summary
    report += ["## Phase 1.5 — Coordinate Ground-Truth Validation",
               f"- Monuments validated against Wikipedia/Wikidata: "
               f"**{cv['n_found']:,} / {cv['n_unique_monuments']:,}** "
               f"({cv['found_pct']:.1f}%)",
               f"- Sources: Wikipedia={cv['source_wikipedia']}, "
               f"Wikidata={cv['source_wikidata']}"]
    if "MAE_km" in cv:
        report += [
            f"- MAE vs ground truth: **{cv['MAE_km']:.3f} km**",
            f"- Median error: **{cv['median_km']:.3f} km**",
            f"- Within 1 km: **{cv['pct_within_1km']:.1f}%** | "
            f"Within 10 km: **{cv['pct_within_10km']:.1f}%** | "
            f"Within 50 km: **{cv['pct_within_50km']:.1f}%**",
            f"- Suspect matches (>100 km): **{cv['pct_over_100km']:.1f}%**",
        ]
    report.append(f"- Outputs: `{COORD_VAL_DIR}/` "
                  f"(validation CSV, worst-errors CSV, metrics JSON, 2 figures)\n")
if benchmark_summary is not None:
    report += ["## Phase 2 — Retrieval Benchmark",
               "| Model | " + " | ".join(benchmark_summary["METRICS"]) + " |",
               "|---|" + "---|" * len(benchmark_summary["METRICS"])]
    for lab in benchmark_summary["LABELS"]:
        row = [lab] + [f"{benchmark_summary['overall'][lab][m]['mean']:.4f}"
                       for m in benchmark_summary["METRICS"]]
        report.append("| " + " | ".join(row) + " |")
    sig_lines = [s for s in benchmark_summary["sig_table"] if s["significant"]]
    report += [f"\n**Significant pairs (Holm p<0.05):** "
               f"{len(sig_lines)} of {len(benchmark_summary['sig_table'])}"]
    # Geo-error summary table
    if benchmark_summary.get("geo_error"):
        ge = benchmark_summary["geo_error"]
        geo_labs = [l for l in benchmark_summary["LABELS"] if "MAE_geo_km" in ge.get(l,{})]
        if geo_labs:
            report += ["\n### Geo-error metrics (coordinate-aware)",
                       "| Model | MAE (km) | MedAE (km) | GeoAcc@10km | GeoAcc@161km | Geo-wMRR |",
                       "|---|---:|---:|---:|---:|---:|"]
            for lab in geo_labs:
                g = ge[lab]
                report.append(
                    f"| {lab} | {g['MAE_geo_km']:.1f} | {g['MedAE_geo_km']:.1f} | "
                    f"{g.get('GeoAcc@10km',0)*100:.1f}% | "
                    f"{g.get('GeoAcc@161km',0)*100:.1f}% | "
                    f"{g.get('geo_weighted_MRR',float('nan')):.4f} |")
    report.append(f"\n- Outputs: `{BENCH_DIR}/` (tables 3–7 + figures + .tex + JSON/CSV)\n")
if finetune_summary is not None:
    fs = finetune_summary
    report += ["## Phase 3 — Fine-tuning Experiment",
               f"- Fine-tuned **{fs['ft_label']}** on {fs['N_train']:,} pairs, "
               f"tested on {fs['N_test']:,}.",
               f"- Improvement over best zero-shot baseline "
               f"(**{fs['best_zs_label']}**) on the same test pool: "
               f"**Δ MRR = {fs['delta_MRR']:+.4f}**",
               "- Per-group MRR (fine-tuned):"]
    for g, gm in fs["ft_group_metrics"].items():
        if gm.get("N", 0) == 0: continue
        report.append(f"  - {g} (n={gm['N']}): "
                      f"MRR = {gm['MRR']['mean']:.4f} "
                      f"[{gm['MRR']['ci_lo']:.4f}, {gm['MRR']['ci_hi']:.4f}]")
    report.append(f"- Outputs: `table6_finetune_vs_zeroshot`, "
                  f"`fig_finetune_improvement`, `finetune_metrics.json`, `table6.tex`")
    if fs.get("geo_error_finetuned") and "MAE_geo_km" in fs["geo_error_finetuned"]:
        g = fs["geo_error_finetuned"]
        report.append(f"- **Geo-error (fine-tuned on test split):** "
                      f"MAE={g['MAE_geo_km']:.1f} km | "
                      f"GeoAcc@161km={g.get('GeoAcc@161km',0)*100:.1f}% | "
                      f"Geo-wMRR={g.get('geo_weighted_MRR',float('nan')):.4f}")
    report.append("")
report += ["## Troubleshooting — CUDA / cuBLAS errors in Phase 2/3",
           "If a model failed with `CUBLAS_STATUS_NOT_SUPPORTED`, "
           "`CUDA error`, or `out of memory`, the pipeline auto-retried it on "
           "CPU and, failing that, skipped it so the rest of the benchmark "
           "still completed. To recover that model:",
           "1. **Restart the runtime** (Runtime → Restart session) and re-run "
           "— the most common cause is a corrupted GPU state from an earlier crash.",
           "2. **Lower the batch size**: set `CONFIG['batch_size'] = 8` (or 4). "
           "cuBLAS heuristic failures are often shape/memory related.",
           "3. **Update libraries once**: "
           "`!pip install -U torch transformers sentence-transformers` "
           "(version mismatches trigger this on Colab).",
           "4. **mGTE-base specifically** uses custom CUDA code via "
           "`trust_remote_code`; if only it fails, either keep the automatic "
           "CPU fallback or drop it from `CONFIG['models']`.\n"]
report += ["## Output tree",
           f"```",
           f"{ROOT}/",
           f"├── descriptive/        Phase 1: fig1–fig8, JSON, MD",
           f"├── coord_validation/   Phase 1.5: validation CSV, worst-errors,",
           f"│                       metrics JSON, 2 figures (histogram+map)",
           f"├── benchmark/          Phase 2: tables 3–7, figures, LaTeX, JSON",
           f"├── cache/              Embedding cache (re-runs are fast)",
           f"└── report.md           This file",
           f"```"]
with open(os.path.join(ROOT, "report.md"), "w") as f:
    f.write("\n".join(report))
print(f"\nAll outputs in: {ROOT}/")
print(f"  ├── descriptive/")
print(f"  ├── benchmark/")
print(f"  ├── cache/")
print(f"  └── report.md")
