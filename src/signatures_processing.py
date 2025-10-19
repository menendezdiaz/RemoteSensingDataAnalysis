import numpy as np
from pathlib import Path

# ------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------

def apply_mask(wavelength, reflectance):
    """
    Apply cleaning mask to spectral data:
    - Removes atmospheric absorption bands (1350–1450, 1810–1970, 2320+ nm)
    - Removes invalid reflectance values (<0 or >1)
    """
    mask = (
        ((wavelength >= 1350) & (wavelength <= 1450)) |
        ((wavelength >= 1810) & (wavelength <= 1970)) |
        (wavelength >= 2320)
    )
    reflectance[mask] = np.nan
    reflectance[reflectance > 1] = np.nan
    reflectance[reflectance < 0] = np.nan
    return reflectance


# ------------------------------------------------------
# LOAD SIGNATURES
# ------------------------------------------------------

def load_signatures_for_object(folder, base_name, clean=True):
    """
    Loads all files for a given object (e.g. asphalt1.txt, asphalt2.txt, ...)
    Returns:
        wavelength (array)
        mean reflectance (array)
    If clean=True, applies the mask (band removal).
    """
    files = sorted(folder.glob(f'{base_name}*.txt'))
    if not files:
        print(f'⚠️ No files found for {base_name}')
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

    mean_refl = np.nanmean(signatures, axis=0)
    return wl, mean_refl
