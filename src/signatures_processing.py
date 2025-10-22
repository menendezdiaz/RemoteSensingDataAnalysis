import numpy as np
from pathlib import Path




# --------------------------------------------------
# DEFINE ATMOSPHERIC ABSORPTION BANDS (GLOBAL)
# --------------------------------------------------

ATMOSPHERIC_BANDS = [
    (1350, 1450),  # Water vapor absorption
    (1810, 1970),  # Water vapor absorption
    (2320, 2500)   # Beyond useful range
]

def get_atmospheric_bands():
    """Returns the predefined atmospheric absorption wavelength ranges."""
    return ATMOSPHERIC_BANDS



# ------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------

def apply_mask(wavelength, reflectance):
    """
    Apply cleaning mask to spectral data:
    - Removes atmospheric absorption bands (1350–1450, 1810–1970, 2320+ nm)
    - Removes invalid reflectance values (<0 or >1)
    """
    mask = np.zeros_like(wavelength, dtype=bool)
    for start, end in ATMOSPHERIC_BANDS:
        mask |= (wavelength >= start) & (wavelength <= end)

    reflectance[mask] = np.nan
    reflectance[reflectance > 1] = np.nan
    reflectance[reflectance < 0] = np.nan
    return reflectance



# ------------------------------------------------------
# LOAD SIGNATURES
# ------------------------------------------------------

def load_signatures_for_object(folder, base_name, clean=True, return_std=False):
    """
    Loads all files for a given object (e.g. asphalt1.txt, asphalt2.txt, ...)
    Returns:
        wavelength (array)
        mean reflectance (array)
        std reflectance (array) - only if return_std=True
    If clean=True, applies the mask (band removal).
    """
    files = sorted(folder.glob(f'{base_name}*.txt'))
    if not files:
        print(f'⚠️ No files found for {base_name}')
        if return_std:
            return None, None, None
        return None, None

    signatures = []
    wl = None

    for file in files:
        try:
            data = np.loadtxt(file, skiprows=1)
        except Exception:
            data = np.loadtxt(file, skiprows=1, delimiter='\t')

        wl, refl = data[:, 0], data[:, 1]
        if clean:
            refl = apply_mask(wl, refl)
        signatures.append(refl)

    # --- Handle empty or invalid signature arrays safely ---
    signatures = np.array(signatures)

    # Remove rows that are completely NaN
    if signatures.ndim > 1:
        signatures = signatures[~np.isnan(signatures).all(axis=1)]

    # Check if any valid data remains
    if signatures.size == 0 or np.isnan(signatures).all():
        print(f"⚠️ No valid spectral data found for {base_name}.")
        if return_std:
            return wl, None, None
        return wl, None

    # Compute mean and std safely with proper checks
    if signatures.shape[0] == 0:
        # No valid signatures to process
        print(f"⚠️ No valid signatures found for {base_name}.")
        if return_std:
            return wl, None, None
        return wl, None
    
    # Check if all values are NaN along the axis
    if np.isnan(signatures).all(axis=0).all():
        print(f"⚠️ All spectral data is NaN for {base_name}.")
        if return_std:
            return wl, None, None
        return wl, None

    # Compute mean and std safely with additional checks
    # Check if we have enough valid data points for each wavelength
    valid_counts = np.sum(~np.isnan(signatures), axis=0)
    
    # Initialize arrays with NaN
    mean_refl = np.full(signatures.shape[1], np.nan)
    std_refl = np.full(signatures.shape[1], np.nan)
    
    # Only compute mean/std where we have valid data
    valid_mask = valid_counts > 0
    if np.any(valid_mask):
        # Compute mean only where we have at least 1 valid sample
        mean_refl[valid_mask] = np.nanmean(signatures[:, valid_mask], axis=0)
        
        if return_std:
            # Only compute std where we have more than 1 valid sample
            # This prevents the "degrees of freedom <= 0" warning
            std_mask = valid_counts > 1
            if np.any(std_mask):
                # Extract only the valid data for std calculation
                valid_signatures = signatures[:, std_mask]
                # Ensure we have at least 2 non-NaN values per wavelength
                for i, col_idx in enumerate(np.where(std_mask)[0]):
                    col_data = valid_signatures[:, i]
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 1:
                        std_refl[col_idx] = np.std(valid_data, ddof=1)
                    else:
                        std_refl[col_idx] = np.nan
    
    # Check if mean calculation resulted in all NaN values
    if np.isnan(mean_refl).all():
        print(f"⚠️ Mean calculation resulted in all NaN values for {base_name}.")
        if return_std:
            return wl, None, None
        return wl, None

    if return_std:
        std_refl = np.nanstd(signatures, axis=0)
        return wl, mean_refl, std_refl

    return wl, mean_refl






def load_wv2_bands(filepath='data/WV2.txt'):
    """
    Loads WorldView-2 spectral response functions.
    Returns:
        wl (array): Wavelengths
        bands (dict): Dictionary with band names and their responses
    """
    from pathlib import Path
    
    # Try different possible paths
    possible_paths = [
        Path(filepath),
        Path('../') / filepath,
        Path(__file__).parent.parent / filepath
    ]
    
    file_path = None
    for p in possible_paths:
        if p.exists():
            file_path = p
            break
    
    if file_path is None:
        print(f"⚠️ WV2 file not found at {filepath}")
        return None, None
    
    data = np.loadtxt(file_path, skiprows=1)
    wl = data[:, 0]
    
    bands = {
        'Coastal Blue': data[:, 1],
        'Blue': data[:, 2],
        'Green': data[:, 3],
        'Yellow': data[:, 4],
        'Red': data[:, 5],
        'Red Edge': data[:, 6],
        'NIR1': data[:, 7],
        'NIR2': data[:, 8]
    }
    
    return wl, bands