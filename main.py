import cv2
import matplotlib.pyplot as plt
import numpy as np
import os # library para cargar carpetas del
import random # library para generar un numero random
import scipy.ndimage as ndi

from skimage import io

# Ruta a la carpeta con imágenes ruidosas
# ajustar el path para enviar, que sea parametrizable
ruta_low = os.path.join('snr_7_binning_2', 'train', 'low')

# Listar todas las imágenes en la carpeta
imagenes = os.listdir(ruta_low)

# Seleccionar 10 imágenes al azar
imagenes_random = random.sample(imagenes, 10)
print("Imágenes seleccionadas:", imagenes_random)

# PASO 2:  Carga y Visualización de Imagenes
plt.figure(figsize=(16,8))

luts = ['gray', 'magma', 'viridis', 'plasma']

#TODO: ALL LUPS x 1 IMAGE

for i, nombre_img in enumerate(imagenes_random):
    img = io.imread(os.path.join(ruta_low, nombre_img))
    lut_random = random.choice(luts)
    plt.subplot(2, 5, i+1)  # 2 filas x 5 columnas
    plt.imshow(img, cmap=lut_random)
    plt.title(f"Imagen {i+1} con LUT: {lut_random}")
    plt.axis('off')

plt.tight_layout()  # cuidar de overlaping de imagenes
plt.show()


# PASO 3: Calculo y Comparacion de Histogramas
for i, nombre_img in enumerate(imagenes_random):
    img = io.imread(os.path.join(ruta_low, nombre_img))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Imagen
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Imagen {i+1}')
    axes[0].axis('off')
    
    # Histograma
    axes[1].hist(img.ravel(), bins=256, histtype='step', color='black')
    axes[1].set_title(f'Histograma Imagen {i+1}')
    axes[1].set_xlabel('Nivel de Gris')
    axes[1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()

# PASO 4: ELIMINACION DE RUIDOS Y FILTROS 
for i, nombre_img in enumerate(imagenes_random):
    img = io.imread(os.path.join(ruta_low, nombre_img))
    
    # Filtro promedio
    img_mean = ndi.uniform_filter(img, size=5)
    
    # Filtro mediana
    img_median = cv2.medianBlur(img, ksize=5)
    
    # Filtro Gaussiano
    img_gaussian = ndi.gaussian_filter(img, sigma=1.5)
    
    # Mostrar resultados
    plt.figure(figsize=(16,8))
    
    plt.subplot(1,4,1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Imagen {i} Filtro: Original")
    plt.axis('off')
    
    plt.subplot(1,4,2)
    plt.imshow(img_mean, cmap='gray')
    plt.title(f"Imagen {i} Filtro: Promedio")
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.imshow(img_median, cmap='gray')
    plt.title(f"Imagen {i} Filtro: Mediana")
    plt.axis('off')
    
    plt.subplot(1,4,4)
    plt.imshow(img_gaussian, cmap='gray')
    plt.title(f"Imagen {i} Filtro: Gaussiano")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# PASO 5: OPERACIONES MATEMATICAS

multiplicador = 1.5  # para aumentar contraste
suma_constante = 30  # para aumentar brillo

for index, nombre_img in enumerate(imagenes_random, start=1):
    img = io.imread(os.path.join(ruta_low, nombre_img))
    
    # Aplicar realce
    img_multiplicada = np.clip(img * multiplicador, 0, 255).astype(np.uint8)
    img_sumada = np.clip(img + suma_constante, 0, 255).astype(np.uint8)
    
    # Mostrar resultados
    plt.figure(figsize=(16,6))
    
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Imagen {index} - Original")
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(img_multiplicada, cmap='gray')
    plt.title(f"Imagen {index} - Contraste x{multiplicador}")
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(img_sumada, cmap='gray')
    plt.title(f"Imagen {index} - Brillo +{suma_constante}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()