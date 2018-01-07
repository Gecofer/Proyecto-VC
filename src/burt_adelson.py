import cv2
import numpy as np

from math import floor, sqrt
from util import show

# Implementación: https://github.com/stheakanath/multiresolutionblend/blob/master/main.py

def burt_adelson(imgA, imgB, mask):
    """Algoritmo de Burt-Adelson"""

    gaussian_mask = compute_gaussian_pyramid(mask, levels=4)

    # calculamos la piramide laplaciana de las imagenes
    # A y B
    lAs = compute_laplacian_pyramid(imgA.astype(np.float64)/255)
    lBs = compute_laplacian_pyramid(imgB.astype(np.float64)/255)
    
    lSs = [
        l
    ]
    for lA, lB, GR in zip(lAs, lBs, gaussian_mask):
        lSs.append(lA * GR + lB * (1 - GR))
        

    # aqui habria que recomponer las componentes de la laplaciana
    # lSs en una sola imagen

    return lSs

'''
    # creamos la imagen que contendrá la
    # combinación de la imagen A y B
    recomponer_image = np.empty(imgA.shape)
    
    # reconstruimos la imagen desde el nivel 
    # n hasta el nivel 0
    for lS in lSs:
        recomponer_image.append(lS + (lS.shape[0], lS.shape[1]))
        
    imagen_final = recomponer_image
	
	# normalizamos la imagen
	
	return imagen_final
'''

def compute_laplacian_pyramid(img, levels=4):
    g_pyramid = compute_gaussian_pyramid(img, levels)
    pyramid = []
    
    for imgIzq, imgDer in zip(g_pyramid, g_pyramid[1:]):
        height, width = imgIzq.shape[:2]

        pyramid.append(
            np.float64(cv2.addWeighted(
                np.uint8(imgIzq * 255), 1,
                np.uint8(cv2.pyrUp(imgDer)[:height, :width] * 255), -1,
                0
            ))/255
        )

    return pyramid

def compute_gaussian_pyramid(img, levels=4):
    pyramid = [img]
    
    downsampled = img
    for _ in range(levels):
        downsampled = cv2.pyrDown(downsampled)
        pyramid.append(downsampled)

    return pyramid
