"""
GeoQueryCA — coordinate validation v3 (one-file version for Colab or any PC)
=============================================================================
Same method and code as validate_coords3.py + make_coord_fig.py:
every unique landmark of GEOQUERYCA_DATASET_v1.1.csv is looked up in
Wikidata (P625) and, if not found, in GeoNames, with the country checked by
ISO 3166 code for both sources; network errors are retried. Then the summary
statistics (paper §3.2, Table 2) and Figure 2 are produced.

How to run — no Google Drive or manual upload needed
  • Google Colab: paste this file into one cell and run it.
  • Any computer: python coord_validation_v3_colab.py
The dataset is used from the current folder if present; otherwise it is
downloaded automatically from the GitHub repository. Its SHA-256 checksum is
verified so that the run always uses the released file.

The run takes about 2.5 hours (GeoNames allows ~1 request per 4 s). If it is
interrupted, run it again in the same folder/session: it resumes from
coord_validation_v3full.csv.

Outputs (in the current folder; in Colab they are also downloaded):
  coord_validation_v3full.csv, validation_v3full_summary.txt,
  fig_coord_qa.pdf, fig_coord_qa.png
"""
import subprocess, sys, os, hashlib, urllib.request
try:
    import pycountry  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycountry"], check=True)

EXPECTED_SHA256 = "3fe343de307bba19107a42d4aafea6e143acf49d88fda864194647f6aea8a4cf"
DATA_NAME = "GEOQUERYCA_DATASET_v1.1.csv"
DATA_URL = ("https://raw.githubusercontent.com/Geotagging-Research/GeoQueryCA/"
            "main/GEOQUERYCA_DATASET_v1.1.csv")

try:
    from google.colab import files as _files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("Working folder:", os.getcwd())

# ── dataset: local copy, or download from GitHub ─────────────────────────────
if not os.path.exists(DATA_NAME):
    print(f"Downloading {DATA_NAME} from GitHub …")
    urllib.request.urlretrieve(DATA_URL, DATA_NAME)
sha = hashlib.sha256(open(DATA_NAME, "rb").read()).hexdigest()
if sha != EXPECTED_SHA256:
    sys.exit(f"WRONG DATASET (sha256 {sha[:16]}…, expected {EXPECTED_SHA256[:16]}…). "
             f"Delete {DATA_NAME} from {os.getcwd()} and run again to download the released file.")
print(f"Dataset OK: {DATA_NAME} (sha256 {sha[:16]}…)")

import math, os, re, time, datetime
import numpy as np, pandas as pd, requests
import pycountry

CSV = DATA_NAME
OUT = "coord_validation_v3full.csv"
GEONAMES_USER = "sahoomadhusmita"
GN_DELAY, WD_DELAY = 4.0, 1.0
# Wikimedia asks automated clients to identify themselves with a contact URL
UA = {"User-Agent": "GeoQueryCA-validation/3.0 (https://github.com/Geotagging-Research/GeoQueryCA)"}
WD = "https://www.wikidata.org/w/api.php"

# ISO codes for names pycountry does not resolve directly
ISO_OVERRIDE = {"Russia": "RU", "Democratic Republic of the Congo": "CD",
                "Palestine": "PS", "Vatican City": "VA", "Türkiye": "TR",
                "Turkey": "TR", "South Korea": "KR", "Laos": "LA",
                "Antarctica": "AQ"}


def iso2(country):
    c = str(country).strip()
    if c in ISO_OVERRIDE: return ISO_OVERRIDE[c]
    try: return pycountry.countries.lookup(c).alpha_2
    except LookupError:
        try: return pycountry.countries.search_fuzzy(c)[0].alpha_2
        except LookupError: return None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def name_variants(name):                       # identical to v2
    n = str(name).strip()
    vs = [n]
    no_paren = re.sub(r"\s*\([^)]*\)", "", n).strip()
    if no_paren != n: vs.append(no_paren)
    parts = n.split()
    if len(parts) >= 3: vs.append(" ".join(parts[1:]))
    return list(dict.fromkeys(vs))


class LookupError_(Exception):
    pass


_last = {"wd": 0.0, "gn": 0.0}
_slow = {"wd": 1.0, "gn": 1.0}          # grows when a service asks us to slow down
def get_json(url, params, kind, delay, headers=None):
    """GET with rate limiting and up to 4 attempts.
    Honours HTTP 429 / Retry-After and slows down for the rest of the run.
    Raises LookupError_ with the status code if every attempt fails."""
    why = ""
    for attempt in range(4):
        wait = delay * _slow[kind] - (time.time() - _last[kind])
        if wait > 0: time.sleep(wait)
        _last[kind] = time.time()
        try:
            r = requests.get(url, params=params, timeout=30, headers=headers)
            if r.status_code == 200:
                try:
                    j = r.json()
                    if isinstance(j, dict) and "error" in j and kind == "wd":
                        why = "api-error " + str(j["error"].get("code"))
                    else:
                        return j
                except ValueError:
                    why = "non-JSON reply"
            else:
                why = f"HTTP {r.status_code}"
                if r.status_code in (429, 503):
                    ra = r.headers.get("Retry-After")
                    _slow[kind] = min(_slow[kind] * 2, 8)
                    time.sleep(float(ra) if ra and ra.isdigit() else 30)
        except requests.RequestException as e:
            why = type(e).__name__
        time.sleep(10 * (attempt + 1))
    raise LookupError_(why)


_qid_iso = {}
def qid_to_iso(qid):
    if qid in _qid_iso: return _qid_iso[qid]
    j = get_json(WD, {"action": "wbgetclaims", "entity": qid, "property": "P297",
                      "format": "json"}, "wd", WD_DELAY, UA)
    cl = j.get("claims", {}).get("P297", [])
    code = cl[0]["mainsnak"]["datavalue"]["value"] if cl else None
    _qid_iso[qid] = code
    return code


def wikidata_lookup(name, cc):
    for q in name_variants(name):
        j = get_json(WD, {"action": "wbsearchentities", "search": q, "language": "en",
                          "format": "json", "limit": 5}, "wd", WD_DELAY, UA)
        for hit in j.get("search", []):
            e = get_json(WD, {"action": "wbgetentities", "ids": hit["id"],
                              "props": "claims|labels", "languages": "en",
                              "format": "json"}, "wd", WD_DELAY, UA)
            ent = e.get("entities", {}).get(hit["id"], {})
            claims = ent.get("claims", {})
            if "P625" not in claims: continue
            if cc and "P17" in claims:            # country check (when P17 exists)
                countries = {qid_to_iso(c["mainsnak"]["datavalue"]["value"]["id"])
                             for c in claims["P17"]
                             if c["mainsnak"].get("datavalue")}
                if cc not in countries: continue
            v = claims["P625"][0]["mainsnak"].get("datavalue", {}).get("value")
            if not v: continue
            label = ent.get("labels", {}).get("en", {}).get("value", hit.get("label"))
            return (v["latitude"], v["longitude"], "wikidata", hit["id"], label)
    return None


def geonames_lookup(name, cc):
    for q in name_variants(name):
        j = get_json("http://api.geonames.org/searchJSON",
                     {"q": q, "maxRows": 5, "username": GEONAMES_USER}, "gn", GN_DELAY)
        if "status" in j:
            raise LookupError_("GeoNames: " + j["status"].get("message", ""))
        for g in j.get("geonames", []):
            if cc and g.get("countryCode") != cc: continue
            return (float(g["lat"]), float(g["lng"]), "geonames",
                    str(g.get("geonameId")), g.get("name"))
    return None


df = pd.read_csv(CSV, encoding="utf-8-sig").rename(columns={"Landmark": "Monument"})
mons = df.drop_duplicates("Monument")[["Monument", "Latitude", "Longitude", "Country"]] \
         .reset_index(drop=True)
print(f"Unique monuments: {len(mons)}   started {datetime.datetime.now():%Y-%m-%d %H:%M}",
      flush=True)

done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
if len(done):                         # retry earlier lookup errors on resume
    done = done[done["note"] != "lookup_error"]
seen = set(done["Monument"]) if len(done) else set()
rows = done.to_dict("records") if len(done) else []
if seen: print(f"Resuming: {len(seen)} already processed", flush=True)

for _, r in mons.iterrows():
    if r["Monument"] in seen: continue
    cc = iso2(r["Country"])
    row = {"Monument": r["Monument"], "Country": r["Country"], "iso2": cc,
           "source": None, "ref_id": None, "ref_label": None,
           "ref_lat": None, "ref_lon": None, "distance_km": None, "note": None}
    try:
        hit = wikidata_lookup(r["Monument"], cc) or geonames_lookup(r["Monument"], cc)
        if hit:
            lat, lon, src, rid, lab = hit
            row.update(source=src, ref_id=rid, ref_label=lab, ref_lat=lat, ref_lon=lon,
                       distance_km=haversine(float(r["Latitude"]), float(r["Longitude"]),
                                             lat, lon))
        else:
            row["note"] = "no_match"
    except LookupError_ as e:
        row["note"] = "lookup_error"
        print(f"  lookup error for {r['Monument']}: {e}", flush=True)
        n_err = sum(1 for x in rows if x.get("note") == "lookup_error") + 1
        if n_err >= 20 and n_err > 0.2 * (len(rows) + 1):
            pd.DataFrame(rows + [row]).to_csv(OUT, index=False)
            raise SystemExit(f"Stopping: {n_err} lookup errors so far (last: {e}). "
                             "Fix the connection, then re-run — the script resumes.")
    rows.append(row)
    if len(rows) % 25 == 0:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  {len(rows)}/{len(mons)} processed (checkpoint saved)", flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

# Summary — same definitions as make_coord_fig.py (case-insensitive dedup)
o = out.copy()
o["key"] = o["Monument"].str.strip().str.lower()
o = o.sort_values("distance_km").drop_duplicates("key")
m = o.dropna(subset=["distance_km"])
d = m["distance_km"].astype(float).values
dc = d[d <= 100]
_summary = []
def _p(s=""):
    print(s); _summary.append(s)
_p(f"\n===== VALIDATION SUMMARY (v3 full, {datetime.date.today()}) =====")
_p(f"Matched: {len(m)}/{len(o)} ({len(m)/len(o)*100:.1f}%)")
for src, cnt in m["source"].value_counts().items():
    _p(f"  {src}: {cnt}")
_p(f"Lookup errors (retried, unresolved): {(o['note'] == 'lookup_error').sum()}")
_p(f"Median: {np.median(d):.3f} km | MAE: {np.mean(d):.1f} km")
_p(f"Gross failures (>100 km): {(d > 100).mean()*100:.1f}%")
_p(f"-- Excluding gross failures (n={len(dc)}) --")
for thr in [1, 5, 10, 25, 50]:
    _p(f"  within {thr:2d} km: {(dc <= thr).mean()*100:.1f}%")
_p(f"Saved to {OUT}")

with open("validation_v3full_summary.txt", "w") as f:
    f.write("\n".join(_summary) + "\n")

# ── Figure 2 (identical to make_coord_fig.py) ───────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
C_NAVY, C_SKY, C_RED = "#1F3B70", "#5B8DB8", "#C0392B"
with plt.rc_context({"font.family": "serif", "font.size": 11}):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    dcs = np.sort(dc)
    ax.plot(dcs, np.arange(1, len(dcs) + 1) / len(dcs) * 100, color=C_NAVY, lw=2)
    ax.set_xscale("log"); ax.set_xlim(0.01, 120); ax.set_ylim(0, 102)
    ax.set_xlabel("Coordinate discrepancy (km, log scale)")
    ax.set_ylabel("Cumulative % of matched monuments")
    for thr in [1, 5, 10, 25, 50]:
        pct = (dcs <= thr).mean() * 100
        ax.axvline(thr, color=C_SKY, ls="--", lw=1)
        ax.annotate("%.1f%%\n≤%d km" % (pct, thr), xy=(thr, pct),
                    xytext=(thr * 1.12, max(pct - (28 if thr == 50 else 14), 4)),
                    fontsize=8, color=C_NAVY)
    med = float(np.median(d))
    ax.axvline(med, color=C_RED, ls=":", lw=1.2)
    ax.annotate("median %.2f km" % med, xy=(med, 5), xytext=(med * 1.3, 5),
                fontsize=8, color=C_RED)
    ax.grid(alpha=0.25, which="both")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig("fig_coord_qa.pdf", bbox_inches="tight")
    fig.savefig("fig_coord_qa.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
print("saved fig_coord_qa.pdf/.png")

if IN_COLAB:
    for f in ["coord_validation_v3full.csv", "validation_v3full_summary.txt",
              "fig_coord_qa.pdf", "fig_coord_qa.png"]:
        _files.download(f)
print("ALL DONE — outputs in", os.getcwd())
