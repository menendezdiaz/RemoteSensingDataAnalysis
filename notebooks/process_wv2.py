"""
Script para procesar firmas espectrales y simular bandas de WorldView-2
Ejecutar desde: notebooks/
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
import pandas as pd
import sys

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

# Obtener el directorio del script actual
if '__file__' in globals():
    script_dir = Path(__file__).resolve().parent
else:
    script_dir = Path.cwd()

# Directorio raíz del proyecto (un nivel arriba de notebooks)
root_dir = script_dir.parent if script_dir.name == 'notebooks' else script_dir

# Añadir al path para importar módulos
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Definir rutas de datos
data_folder = root_dir / 'data' / 'field'
srf_file = root_dir / 'data' / 'WV2.txt'

print(f"Directorio raíz: {root_dir}")
print(f"Carpeta de datos: {data_folder}")
print(f"Archivo SRF: {srf_file}")
print()

# Importar módulo de procesamiento
try:
    from src.signatures_processing import load_signatures_for_object
    print("✓ Módulo 'src.signatures_processing' importado correctamente\n")
except ImportError as e:
    print(f"⚠️  Error al importar: {e}")
    print("Asegúrate de que existe el archivo: src/signatures_processing.py\n")
    sys.exit(1)


# ============================================================
# FUNCIONES DE PROCESAMIENTO WV-2
# ============================================================

def load_wv2_srf(file_path):
    """
    Carga las Funciones de Respuesta Espectral (SRF) de WorldView-2.
    """
    df = pd.read_csv(file_path, sep='\t')
    
    band_columns = {
        'WV2 Coastal Blue': 'Coastal',
        'WV2 Blue': 'Blue', 
        'WV2 Green': 'Green',
        'WV2 Yellow': 'Yellow',
        'WV2 Red': 'Red',
        'WV2 RedEdge': 'Red Edge',
        'WV2 NIR1': 'NIR1',
        'WV2 NIR2': 'NIR2'
    }
    
    bands_srf = {}
    wavelength = df['WL(nm)'].values
    
    for col_name, band_name in band_columns.items():
        response = df[col_name].values
        
        valid_mask = response > 0.01
        if np.any(valid_mask):
            center = np.average(wavelength[valid_mask], weights=response[valid_mask])
        else:
            center = wavelength[np.argmax(response)]
        
        bands_srf[band_name] = {
            'wavelength': wavelength,
            'response': response,
            'center': center
        }
    
    return bands_srf


def convolve_signature_with_band(wavelength, reflectance, band_wl, band_response):
    """
    Convoluciona una firma espectral con la función de respuesta de una banda.
    
    Formula: R_band = ∫ R(λ) * SRF(λ) dλ / ∫ SRF(λ) dλ
    """
    refl_interp = interp1d(wavelength, reflectance, kind='linear', 
                           bounds_error=False, fill_value=np.nan)
    
    refl_at_srf = refl_interp(band_wl)
    valid_mask = (~np.isnan(refl_at_srf)) & (band_response > 1e-6)
    
    if not np.any(valid_mask):
        return np.nan
    
    numerator = np.trapz(
        refl_at_srf[valid_mask] * band_response[valid_mask], 
        band_wl[valid_mask]
    )
    denominator = np.trapz(band_response[valid_mask], band_wl[valid_mask])
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator


def simulate_wv2_bands(wavelength, reflectance, bands_srf):
    """
    Simula los valores de reflectancia para todas las bandas de WorldView-2.
    """
    band_values = {}
    
    for band_name, band_data in bands_srf.items():
        value = convolve_signature_with_band(
            wavelength, reflectance,
            band_data['wavelength'],
            band_data['response']
        )
        band_values[band_name] = value
    
    return band_values


def plot_signature_with_wv2_bands(wavelength, reflectance, band_values, 
                                  bands_srf, title=""):
    """
    Grafica la firma espectral con las bandas WV-2 y valores simulados.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Panel superior: Firma + SRF
    ax1.plot(wavelength, reflectance, 'k-', linewidth=2.5, 
            label='Firma espectral', zorder=10)
    
    band_colors = {
        'Coastal': '#4B0082', 'Blue': '#0000FF', 'Green': '#00FF00',
        'Yellow': '#FFFF00', 'Red': '#FF0000', 'Red Edge': '#8B0000',
        'NIR1': '#CD5C5C', 'NIR2': '#8B4513'
    }
    
    max_refl = np.nanmax(reflectance) if not np.all(np.isnan(reflectance)) else 1.0
    
    for band_name, band_data in bands_srf.items():
        color = band_colors.get(band_name, 'gray')
        srf_normalized = band_data['response'] / np.max(band_data['response'])
        scaled_response = srf_normalized * max_refl * 0.4
        
        ax1.fill_between(band_data['wavelength'], 0, scaled_response, 
                        alpha=0.25, color=color, label=f"{band_name}")
        ax1.plot(band_data['wavelength'], scaled_response, 
                color=color, linewidth=1.2, alpha=0.8)
    
    ax1.set_xlabel('Longitud de onda [nm]', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Reflectancia', fontsize=13, fontweight='bold')
    ax1.set_title(f'Firma espectral con bandas WV-2 - {title}', 
                 fontsize=15, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(350, 1100)
    ax1.set_ylim(0, max_refl * 1.05)
    
    # Panel inferior: Barras con valores
    band_names = list(band_values.keys())
    band_vals = [band_values[name] for name in band_names]
    colors = [band_colors.get(name, 'gray') for name in band_names]
    
    bars = ax2.bar(band_names, band_vals, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, band_vals):
        height = bar.get_height()
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Reflectancia simulada', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Banda WV-2', fontsize=13, fontweight='bold')
    ax2.set_title('Valores de banda convolucionados', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    max_val = np.nanmax(band_vals) if not np.all(np.isnan(band_vals)) else 1.0
    ax2.set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    return fig


def create_spectral_library(data_folder, categories, bands_srf, output_csv):
    """
    Crea una biblioteca espectral completa con valores WV-2.
    """
    results = {}
    
    print("Creando biblioteca espectral WV-2...\n")
    
    for category, objects in categories.items():
        print(f"Categoría: {category}")
        for obj in objects:
            wl, refl = load_signatures_for_object(data_folder, obj, clean=True)
            
            if wl is not None and refl is not None:
                band_values = simulate_wv2_bands(wl, refl, bands_srf)
                results[obj] = band_values
                print(f"  ✓ {obj}")
        print()
    
    df = pd.DataFrame(results).T
    df.index.name = 'Object'
    
    category_map = {}
    for cat, objs in categories.items():
        for obj in objs:
            category_map[obj] = cat
    
    df.insert(0, 'Category', df.index.map(category_map))
    
    df.to_csv(output_csv)
    print(f"✓ Biblioteca espectral guardada en: {output_csv}")
    print(f"  Objetos procesados: {len(df)}")
    print(f"  Bandas: {len(df.columns) - 1}\n")
    
    return df


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    
    # Verificar que existen los archivos necesarios
    if not srf_file.exists():
        print(f"❌ Error: No se encuentra el archivo {srf_file}")
        print("   Asegúrate de que existe: data/WV2.txt")
        sys.exit(1)
    
    if not data_folder.exists():
        print(f"❌ Error: No se encuentra la carpeta {data_folder}")
        sys.exit(1)
    
    # Cargar funciones de respuesta espectral
    print("Cargando funciones de respuesta espectral de WV-2...")
    bands_srf = load_wv2_srf(srf_file)
    print(f"✓ Cargadas {len(bands_srf)} bandas\n")
    
    # Definir categorías y objetos
    categories = {
        'surfaces': ['black_asphalt', 'blue_paint', 'wood_table', 'concrete'],
        'soils': ['soil', 'dry_grass', 'rough_soil'],
        'vegetation': ['lemon_leaf', 'red_flower', 'rosemary', 'jatropha', 
                      'small_flower', 'weed', 'grass']
    }
    
    # ========================================================
    # OPCIÓN 1: Procesar un objeto específico
    # ========================================================
    print("=" * 70)
    print("PROCESANDO OBJETO: lemon_leaf")
    print("=" * 70)
    
    wl, refl = load_signatures_for_object(data_folder, 'lemon_leaf', clean=True)
    
    if wl is not None and refl is not None:
        band_values = simulate_wv2_bands(wl, refl, bands_srf)
        
        print("\nValores de banda WV-2 para 'lemon_leaf':")
        print("-" * 40)
        for band, value in band_values.items():
            print(f"  {band:12s}: {value:.4f}")
        print()
        
        # Visualizar
        fig = plot_signature_with_wv2_bands(wl, refl, band_values, bands_srf, 
                                           title="Lemon Leaf")
        plt.savefig('lemon_leaf_wv2.png', dpi=150, bbox_inches='tight')
        print("✓ Gráfica guardada: lemon_leaf_wv2.png\n")
        plt.show()
    
    # ========================================================
    # OPCIÓN 2: Crear biblioteca espectral completa
    # ========================================================
    print("=" * 70)
    print("CREANDO BIBLIOTECA ESPECTRAL")
    print("=" * 70)
    
    output_csv = root_dir / 'spectral_library_wv2.csv'
    library_df = create_spectral_library(data_folder, categories, bands_srf, output_csv)
    
    print("\nPrimeras filas de la biblioteca:")
    print(library_df.head())
    print()
    
    # ========================================================
    # OPCIÓN 3: Estadísticas por categoría
    # ========================================================
    print("=" * 70)
    print("ESTADÍSTICAS POR CATEGORÍA")
    print("=" * 70)
    
    for category in categories.keys():
        cat_data = library_df[library_df['Category'] == category]
        print(f"\n{category.upper()}:")
        print(cat_data.drop('Category', axis=1).mean())
    
    print("\n" + "=" * 70)
    print("✓ PROCESAMIENTO COMPLETADO")
    print("=" * 70)