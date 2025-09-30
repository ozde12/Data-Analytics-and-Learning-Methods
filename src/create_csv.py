# =========================
# Dataset Builder + Auditor
# =========================
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy


# ---------- Feature helpers ----------
def _time_features(sig: np.ndarray) -> dict:
    if sig.size == 0:
        return {k: np.nan for k in ["mean","std","min","max","median","iqr","rms","skew","kurtosis"]}
    return {
        "mean": float(np.mean(sig)),
        "std": float(np.std(sig, ddof=1)) if sig.size > 1 else 0.0,
        "min": float(np.min(sig)),
        "max": float(np.max(sig)),
        "median": float(np.median(sig)),
        "iqr": float(np.subtract(*np.percentile(sig, [75, 25]))),
        "rms": float(np.sqrt(np.mean(sig**2))),
        "skew": float(skew(sig, bias=False)) if sig.size > 2 else 0.0,
        "kurtosis": float(kurtosis(sig, fisher=True, bias=False)) if sig.size > 3 else 0.0,
    }

def _spectral_features(sig: np.ndarray, fs: float) -> dict:
    if sig.size == 0 or not np.isfinite(fs) or fs <= 0:
        return {"dom_freq": np.nan, "spec_entropy": np.nan, "spec_energy": np.nan, "spec_centroid": np.nan}
    x = sig - np.mean(sig)
    n = x.size
    if n < 2:
        return {"dom_freq": np.nan, "spec_entropy": np.nan, "spec_energy": np.nan, "spec_centroid": np.nan}
    # Hann window to reduce leakage
    hann = 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/n)
    xw = x * hann
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    spec = np.abs(np.fft.rfft(xw))**2
    if spec.sum() <= 0:
        return {"dom_freq": 0.0, "spec_entropy": 0.0, "spec_energy": 0.0, "spec_centroid": 0.0}

    # Dominant frequency (prefer non-DC)
    spec_no_dc = spec.copy()
    if spec_no_dc.size > 1:
        spec_no_dc[0] = 0.0
    dom_idx = int(np.argmax(spec_no_dc)) if spec_no_dc.sum() > 0 else int(np.argmax(spec))
    dom_freq = float(freqs[dom_idx])

    p = spec / spec.sum()
    return {
        "dom_freq": dom_freq,
        "spec_entropy": float(entropy(p)),              # natural log base
        "spec_energy": float(spec.sum() / n),          # average spectral energy
        "spec_centroid": float((freqs * spec).sum() / spec.sum()),
    }

def _per_axis_and_mag_features(df_sensor: pd.DataFrame, fs: float, prefix: str) -> dict:
    """
    Computes (for x,y,z,mag): mean,std,min,max,median,iqr,rms,skew,kurtosis,
    dom_freq,spec_entropy,spec_energy,spec_centroid + corr_x_y, corr_y_z, corr_x_z.
    All keys are prefixed, e.g., 'acc_x_mean', 'gyro_mag_dom_freq', ...
    """
    # Require x,y,z
    for c in ("x","y","z"):
        if c not in df_sensor.columns:
            raise ValueError(f"missing axis '{c}'")

    x = df_sensor["x"].to_numpy(dtype=float, copy=False)
    y = df_sensor["y"].to_numpy(dtype=float, copy=False)
    z = df_sensor["z"].to_numpy(dtype=float, copy=False)
    mag = np.sqrt(x**2 + y**2 + z**2)

    out = {}
    def add(name: str, sig: np.ndarray):
        t = _time_features(sig)
        s = _spectral_features(sig, fs)
        out.update({f"{prefix}{name}_{k}": v for k, v in t.items()})
        out.update({f"{prefix}{name}_{k}": v for k, v in s.items()})

    for name, sig in (("x",x), ("y",y), ("z",z), ("mag",mag)):
        add(name, sig)

    # Cross-axis correlations
    if (len(x) > 1 and np.std(x, ddof=1) > 0 and
        np.std(y, ddof=1) > 0 and np.std(z, ddof=1) > 0):
        out[f"{prefix}corr_x_y"] = float(np.corrcoef(x, y)[0,1])
        out[f"{prefix}corr_y_z"] = float(np.corrcoef(y, z)[0,1])
        out[f"{prefix}corr_x_z"] = float(np.corrcoef(x, z)[0,1])
    else:
        out[f"{prefix}corr_x_y"] = np.nan
        out[f"{prefix}corr_y_z"] = np.nan
        out[f"{prefix}corr_x_z"] = np.nan

    return out


# ---------- File/CSV utilities ----------
def _find_sensor_file(win_dir: Path, target_name: str) -> Path | None:
    """Case-insensitive lookup for sensor CSV inside a window folder."""
    target = target_name.lower()
    for p in win_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".csv" and p.name.lower() == target:
            return p
    return None

def _read_csv_flex(fpath: Path) -> pd.DataFrame | None:
    """
    Robust CSV reader:
    - tries comma/semicolon/tab
    - strips column whitespace
    - normalizes axis names to x,y,z (accepts X/Y/Z too)
    """
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(fpath, sep=sep)
            df.columns = [str(c).strip() for c in df.columns]
            lower = {c.lower(): c for c in df.columns}
            ren = {}
            for want in ("x","y","z"):
                if want in lower: ren[lower[want]] = want
                elif want.upper() in lower: ren[lower[want.upper()]] = want
            if ren:
                df = df.rename(columns=ren)
            return df
        except Exception:
            continue
    return None


# ---------- AUDIT ----------
def audit_windows(WINDOWED_ROOT: str | Path, activities: list[str] | None = None) -> None:
    """
    Reports count of window folders per activity and why any would be skipped.
    """
    root = Path(WINDOWED_ROOT)
    if activities is None:
        activities = sorted([p.name for p in root.iterdir() if p.is_dir()])

    print("[Audit]")
    total_dirs = total_usable = 0
    for act in activities:
        act_dir = root / act
        if not act_dir.exists():
            print(f"  - {act}: MISSING activity folder")
            continue

        win_dirs = sorted([p for p in act_dir.iterdir() if p.is_dir()])
        expected = len(win_dirs)
        total_dirs += expected

        skipped = 0
        reasons = {"missing_acc":0, "missing_gyro":0, "missing_grav":0, "bad_csv":0, "no_xyz":0}
        for w in win_dirs:
            ok = True
            acc = _find_sensor_file(w, "Accelerometer.csv")
            gyr = _find_sensor_file(w, "Gyroscope.csv")
            gra = _find_sensor_file(w, "Gravity.csv")
            if acc is None: reasons["missing_acc"] += 1; ok = False
            if gyr is None: reasons["missing_gyro"] += 1; ok = False
            if gra is None: reasons["missing_grav"] += 1; ok = False

            if ok:
                for pth in (acc, gyr, gra):
                    df = _read_csv_flex(pth)
                    if df is None:
                        reasons["bad_csv"] += 1; ok = False; break
                    cols = [c.lower() for c in df.columns]
                    if not {"x","y","z"}.issubset(set(cols)):
                        reasons["no_xyz"] += 1; ok = False; break

            if not ok:
                skipped += 1

        usable = expected - skipped
        total_usable += usable
        print(f"  - {act}: expected={expected}, usable={usable}, skipped={skipped}, reasons={reasons}")

    print(f"\nTotal window folders: {total_dirs} | Usable (all three files OK): {total_usable}")


# ---------- DATASET BUILDER ----------
def build_dataset_from_windows(
    WINDOWED_ROOT: str | Path,
    OUT_CSV: str | Path = "dataset.csv",
    activities: list[str] | None = None,
    fs: float = 100.0,
    keep_ids: bool = False,   # set True to keep sample/window identifiers
) -> pd.DataFrame:
    """
    Walks WINDOWED_ROOT/<activity>/<window_folder>/ where each window folder contains:
      - Accelerometer.csv
      - Gyroscope.csv
      - Gravity.csv

    Extracts the full feature set per sensor (prefixed 'acc_', 'gyro_', 'grav_'),
    concatenates into a single row per window, adds label 'activity', and saves to OUT_CSV.

    If keep_ids = False (default), columns 'sample_id', 'win_index', 'window_folder' are dropped.
    """
    root = Path(WINDOWED_ROOT)
    if activities is None:
        activities = sorted([p.name for p in root.iterdir() if p.is_dir()])

    sensor_targets = {
        "acc_":  "Accelerometer.csv",
        "gyro_": "Gyroscope.csv",
        "grav_": "Gravity.csv",
    }

    rows: list[dict] = []
    total = 0

    for act in activities:
        act_dir = root / act
        if not act_dir.exists():
            print(f"[Skip] Activity not found: {act_dir}")
            continue

        print(f"\n[Activity] {act}")
        win_dirs = sorted([p for p in act_dir.iterdir() if p.is_dir()])
        for w in win_dirs:
            total += 1
            row = {"activity": act}
            if keep_ids:
                row["window_folder"] = w.name
                if "_win" in w.name:
                    sid, wid = w.name.rsplit("_win", 1)
                    row["sample_id"] = sid
                    row["win_index"] = wid

            ok = True
            for prefix, fname in sensor_targets.items():
                fpath = _find_sensor_file(w, fname)
                if fpath is None:
                    print(f"    !! Missing {fname} in {w}")
                    ok = False; break
                df = _read_csv_flex(fpath)
                if df is None:
                    print(f"    !! Could not read {fpath}")
                    ok = False; break
                try:
                    feats = _per_axis_and_mag_features(df, fs=fs, prefix=prefix)
                except ValueError as e:
                    print(f"    !! {w.name}: {prefix}{e}")
                    ok = False; break
                row.update(feats)

            if ok:
                rows.append(row)

            if total % 250 == 0:
                print(f"  Processed {total} windows...")

    if not rows:
        print("No rows collected.")
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)

    # Drop identifiers unless requested
    if not keep_ids:
        drop_cols = [c for c in ["sample_id","win_index","window_folder"] if c in df_all.columns]
        if drop_cols:
            df_all = df_all.drop(columns=drop_cols)

    # Put label first
    cols = ["activity"] + sorted([c for c in df_all.columns if c != "activity"])
    df_all = df_all[cols]

    out_csv = Path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False)
    print(f"\n[Done] Wrote {len(df_all)} rows × {df_all.shape[1]} columns to: {out_csv}")
    return df_all


if __name__ == "__main__":

    #OUT_CSV = PROJECT_ROOT / "dataset.csv"
    #audit_windows(WINDOW_BASE)
    #df = build_dataset_from_windows(WINDOW_BASE, OUT_CSV, keep_ids=False)