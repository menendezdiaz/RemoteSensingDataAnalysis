"""
Explorar y visualizar imagen TIFF SATELITAL
"""

import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import numpy as np

# 1. Construir la ruta correctamente
script_dir = Path(__file__).parent
ruta_imagen = script_dir.parent / 'data' / 'satellite' / 'Pin9.tif'

# 2. Abrir la imagen satelital
img = tifffile.imread(ruta_imagen)

# 3. Información detallada
print("=" * 50)
print(f"Shape: {img.shape}")
print(f"Tipo de datos: {img.dtype}")
print(f"Número de dimensiones: {img.ndim}")
print(f"Valores min/max: {img.min()} / {img.max()}")
print("=" * 50)

# 4. Función para normalizar con ajuste de contraste
def normalizar_percentil(banda, percentil_min=2, percentil_max=98):
    """Normaliza usando percentiles para mejor contraste"""
    p_min = np.percentile(banda, percentil_min)
    p_max = np.percentile(banda, percentil_max)
    banda_norm = np.clip((banda - p_min) / (p_max - p_min), 0, 1)
    return banda_norm

# 5. Determinar estructura y visualizar
if img.ndim == 3:
    n_bandas = img.shape[2] if img.shape[2] < img.shape[0] else img.shape[0]
    print(f"Imagen con {n_bandas} bandas detectadas\n")
    
    # OPCIÓN A: Si las bandas están en el tercer eje (altura, ancho, bandas)
    if img.shape[2] < img.shape[0]:
        print("Formato detectado: (altura, ancho, bandas)")
        
        # Mostrar todas las bandas individuales
        fig, axes = plt.subplots(2, min(4, img.shape[2]), figsize=(15, 8))
        if img.shape[2] == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if img.shape[2] > 1 else [axes]
        
        for i in range(min(img.shape[2], 8)):
            if i < len(axes):
                banda = img[:, :, i]
                banda_norm = normalizar_percentil(banda)
                axes[i].imshow(banda_norm, cmap='gray')
                axes[i].set_title(f'Banda {i+1}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Si tiene al menos 3 bandas, intentar composición RGB
        if img.shape[2] >= 3:
            # Probar diferentes combinaciones de bandas
            combinaciones = [
                (0, 1, 2, "Bandas 1-2-3 (RGB estándar)"),
                (2, 1, 0, "Bandas 3-2-1 (RGB invertido)"),
            ]
            
            if img.shape[2] >= 4:
                combinaciones.append((3, 2, 1, "Bandas 4-3-2 (Infrarrojo)"))
            
            fig, axes = plt.subplots(1, len(combinaciones), figsize=(15, 5))
            if len(combinaciones) == 1:
                axes = [axes]
            
            for idx, (r, g, b, titulo) in enumerate(combinaciones):
                red = normalizar_percentil(img[:, :, r])
                green = normalizar_percentil(img[:, :, g])
                blue = normalizar_percentil(img[:, :, b])
                
                rgb = np.dstack((red, green, blue))
                
                axes[idx].imshow(rgb)
                axes[idx].set_title(titulo)
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    # OPCIÓN B: Si las bandas están en el primer eje (bandas, altura, ancho)
    else:
        print("Formato detectado: (bandas, altura, ancho)")
        
        # Mostrar todas las bandas individuales
        fig, axes = plt.subplots(2, min(4, img.shape[0]), figsize=(15, 8))
        axes = axes.flatten() if img.shape[0] > 1 else [axes]
        
        for i in range(min(img.shape[0], 8)):
            if i < len(axes):
                banda = img[i, :, :]
                banda_norm = normalizar_percentil(banda)
                axes[i].imshow(banda_norm, cmap='gray')
                axes[i].set_title(f'Banda {i+1}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Si tiene al menos 3 bandas, intentar composición RGB
        if img.shape[0] >= 3:
            combinaciones = [
                (0, 1, 2, "Bandas 1-2-3"),
                (2, 1, 0, "Bandas 3-2-1"),
            ]
            
            if img.shape[0] >= 4:
                combinaciones.append((3, 2, 1, "Bandas 4-3-2"))
            
            fig, axes = plt.subplots(1, len(combinaciones), figsize=(15, 5))
            if len(combinaciones) == 1:
                axes = [axes]
            
            for idx, (r, g, b, titulo) in enumerate(combinaciones):
                red = normalizar_percentil(img[r, :, :])
                green = normalizar_percentil(img[g, :, :])
                blue = normalizar_percentil(img[b, :, :])
                
                rgb = np.dstack((red, green, blue))
                
                axes[idx].imshow(rgb)
                axes[idx].set_title(titulo)
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()

elif img.ndim == 2:
    # Imagen de una sola banda
    print("Imagen de una sola banda (escala de grises)")
    banda_norm = normalizar_percentil(img)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(banda_norm, cmap='gray')
    plt.title('Imagen Satelital (escala de grises)')
    plt.colorbar(label='Valor normalizado', shrink=0.6)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\n✅ Visualización completada")