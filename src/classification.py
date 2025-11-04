import numpy as np
import pandas as pd
from pathlib import Path
import rasterio


WV2_BAND_ORDER = [
    'Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge', 'NIR1', 'NIR2'
]


def _load_reference_signatures(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    available_bands = [c for c in df.columns if c in WV2_BAND_ORDER]
    if 'Object' not in df.columns or not available_bands:
        raise ValueError("CSV must contain 'Object' and at least one WV2 band column")
    return df[['Object'] + available_bands]


def _infer_band_indices(dataset: rasterio.io.DatasetReader) -> dict:
    """
    Try to map WV2 band names to image band indices (1-based). If no metadata is
    available, assume the stored order matches WV2_BAND_ORDER.
    """
    band_desc = []
    try:
        # Some GeoTIFFs store descriptions per band
        for b in range(1, dataset.count + 1):
            band_desc.append(dataset.descriptions[b - 1])
    except Exception:
        band_desc = []

    mapping = {}
    if band_desc and any(band_desc):
        for idx, desc in enumerate(band_desc, start=1):
            if desc in WV2_BAND_ORDER:
                mapping[desc] = idx
    # Fallback to positional order
    if not mapping:
        for i, name in enumerate(WV2_BAND_ORDER, start=1):
            if i <= dataset.count:
                mapping[name] = i
    return mapping


def _scale_to_reflectance(band: np.ndarray) -> np.ndarray:
    """
    Heuristic scaling to [0, 1] if needed. If values look already in [0,1], keep.
    If max is large (e.g., > 2), try dividing by 10000 and clip to [0,1].
    """
    finite = np.isfinite(band)
    if not finite.any():
        return band.astype(np.float32)
    vmax = np.nanmax(band[finite])
    if vmax <= 1.5:
        return band.astype(np.float32)
    scaled = (band / 10000.0).astype(np.float32)
    return np.clip(scaled, 0.0, 1.0)


def classify_min_distance(image_path: str,
                          signatures_csv: str,
                          class_names: list | None = None) -> tuple[np.ndarray, dict, dict]:
    """
    Minimum-distance classifier in WV2 feature space.

    Returns
    -------
    class_map : (H, W) int32 array with class indices (0..K-1), -1 for unknown
    idx_to_class : dict[int,str] index to class name
    class_to_color : dict[str, tuple] RGB color in 0-1 range
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for image classification. Install with: pip install rasterio")
    
    csv_path = Path(signatures_csv)
    df = _load_reference_signatures(csv_path)

    if class_names:
        df = df[df['Object'].isin(class_names)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No classes selected/found in the signatures CSV.")

    with rasterio.open(image_path) as ds:
        band_map = _infer_band_indices(ds)
        # Use only bands present in both CSV and image
        used_bands = [b for b in WV2_BAND_ORDER if b in band_map and b in df.columns]
        if not used_bands:
            raise ValueError("No overlapping WV2 bands between image and signatures.")

        # Read required bands and stack to array (B, H, W)
        bands = []
        for bname in used_bands:
            arr = ds.read(band_map[bname]).astype(np.float32)
            arr = _scale_to_reflectance(arr)
            bands.append(arr)
        img = np.stack(bands, axis=0)  # (B, H, W)

    B, H, W = img.shape

    # Prepare class signature matrix (K, B)
    classes = df['Object'].tolist()
    sig_mat = df[used_bands].to_numpy(dtype=np.float32)

    # Handle missing values per class-band by masking them out in distance
    valid_sig = np.isfinite(sig_mat)
    sig_mat = np.nan_to_num(sig_mat, nan=np.nan)  # keep NaNs for masking

    # Reshape image to (B, N)
    N = H * W
    X = img.reshape(B, N)

    # Compute distances per class in a masked manner
    dists = np.full((len(classes), N), np.inf, dtype=np.float32)
    for k, class_name in enumerate(classes):
        mask_b = valid_sig[k]
        if not mask_b.any():
            continue
        sig_vec = sig_mat[k, mask_b][:, None]        # (Bb, 1)
        Xb = X[mask_b, :]                            # (Bb, N)
        # Euclidean distance
        diff = Xb - sig_vec
        d = np.sqrt(np.sum(diff * diff, axis=0))     # (N,)
        dists[k, :] = d

    # Choose class with minimum distance
    labels = np.argmin(dists, axis=0).astype(np.int32)
    # Pixels where all distances were inf -> unknown = -1
    unknown = ~np.isfinite(dists).any(axis=0)
    labels[unknown] = -1
    class_map = labels.reshape(H, W)

    # Assign stable colors to classes
    base_colors = [
        (0.0, 0.6, 0.2),  # green-ish
        (0.7, 0.7, 0.7),  # gray
        (0.1, 0.4, 0.8),  # blue
        (0.8, 0.1, 0.1),  # red
        (0.9, 0.7, 0.1),  # yellow
        (0.6, 0.3, 0.0),  # brown
        (0.6, 0.0, 0.6),  # purple
        (0.0, 0.7, 0.7),  # cyan
    ]
    class_to_color = {cls: base_colors[i % len(base_colors)] for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for i, cls in enumerate(classes)}

    return class_map, idx_to_class, class_to_color


