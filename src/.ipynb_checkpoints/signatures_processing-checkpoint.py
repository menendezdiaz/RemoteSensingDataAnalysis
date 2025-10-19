import numpy as np
from pathlib import Path


# ------------------------------------------------------
# LIMPIEZA DE DATOS
# ------------------------------------------------------

def apply_mask(wavelength, reflectance):
    """
    Aplica una máscara de limpieza a los datos espectrales.
    """
    mask = (
        ((wavelength >= 1350) & (wavelength <= 1451)) |
        ((wavelength >= 1810) & (wavelength <= 1970)) |
        (wavelength >= 2320)
    )
    reflectance[mask] = np.nan
    reflectance[reflectance > 1] = np.nan
    reflectance[reflectance < 0] = np.nan
    return reflectance


# ------------------------------------------------------
# CARGA Y PROMEDIO DE FIRMAS
# ------------------------------------------------------

def cargar_firmas_por_objeto(folder, nombre_base):
    """
    Carga todos los archivos de un objeto (ej: asphalt1.txt, asphalt2.txt, ...)
    y devuelve la longitud de onda y la media de reflectancia.
    """
    archivos = sorted(folder.glob(f'{nombre_base}*.txt'))
    if not archivos:
        print(f'⚠️ No se encontraron archivos para {nombre_base}')
        return None, None
    
    firmas = []
    for archivo in archivos:
        data = np.loadtxt(archivo, skiprows=1)
        wl, refl = data[:,0], data[:,1]
        refl = apply_mask(wl, refl)
        firmas.append(refl)
    
    mean_refl = np.nanmean(firmas, axis=0)
    return wl, mean_refl
