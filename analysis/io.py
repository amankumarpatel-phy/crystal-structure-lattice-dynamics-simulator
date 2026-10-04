"""Robust XRD data importer with raw-scan/peak-table detection."""
import numpy as np
import pandas as pd

def _read_table(source):
    if hasattr(source, "read"):
        try:
            source.seek(0)
        except Exception:
            pass
        return pd.read_csv(source, comment="#", sep=None, engine="python")
    try:
        return pd.read_csv(source, comment="#", sep=None, engine="python")
    except Exception:
        return pd.read_csv(source, comment="#", header=None, names=["2theta", "intensity"])

def load_xrd_csv(source):
    """
    Returns x, y, x_name, y_name, metadata.

    Recognizes:
      * raw scan: dense 2θ/intensity series
      * peak table: rows containing 2θ, Intensity, and typically hkl/d columns
    """
    df = _read_table(source)
    if df.shape[1] < 2:
        raise ValueError("XRD file must contain at least two columns.")

    cols = {str(c).strip().lower().replace("θ", "theta").replace(" ", ""): c for c in df.columns}

    def find_col(names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    xcol = find_col(["2theta", "twotheta", "2t", "angle", "two_theta"])
    ycol = find_col(["intensity", "counts", "count", "i"])
    if xcol is None or ycol is None:
        xcol, ycol = df.columns[:2]

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        raise ValueError("Not enough numeric XRD points found.")

    hkl_col = find_col(["hkl", "reflection", "miller"])
    d_col = find_col(["d(å)", "d(a)", "d", "dspacing", "dspacingå"])
    has_hkl = hkl_col is not None
    has_d = d_col is not None

    # A generated reflection/peak table commonly has hkl/d columns and is not
    # a uniformly sampled intensity scan. It must not be fed to peak detection.
    unique_step = np.diff(np.sort(x))
    median_step = float(np.median(unique_step)) if unique_step.size else np.nan
    monotonic_fraction = float(np.mean(np.diff(x) >= 0)) if x.size > 1 else 1.0
    duplicate_fraction = 1.0 - (np.unique(x).size / x.size)

    kind = "peak_table" if (has_hkl or has_d) else "raw_scan"
    if kind == "raw_scan":
        # Raw scans should have a reasonably ordered angular axis and more than
        # a few points. We do not impose a universal point-count threshold.
        kind = "raw_scan"

    meta = {
        "kind": kind,
        "n_points": int(x.size),
        "x_column": str(xcol),
        "y_column": str(ycol),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "median_step": median_step,
        "monotonic_fraction": monotonic_fraction,
        "duplicate_fraction": duplicate_fraction,
        "hkl_column": str(hkl_col) if hkl_col is not None else None,
        "d_column": str(d_col) if d_col is not None else None,
        "table": df,
    }

    order = np.argsort(x)
    return x[order], y[order], str(xcol), str(ycol), meta
