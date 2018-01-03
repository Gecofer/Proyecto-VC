import cv2
import numpy as np

from math import floor, sqrt


# Implementación: https://github.com/stheakanath/multiresolutionblend/blob/master/main.py

def burt_adelson(imgA, imgB, maskA, maskB):
    """Algoritmo de Burt-Adelson"""

    # sacamos en que parte de la imagen nueva
    # estan activas ambas imagenes
    shared_mask = maskA * maskB

    # obtenemos la piramide gaussiana de la mascara
    # compartida y de la mascara de A y de B
    gaussian_mask = compute_gaussian_pyramid(shared_mask, levels=4)
    masksA = compute_gaussian_pyramid(maskA, levels=4)
    masksB = compute_gaussian_pyramid(maskB, levels=4)

    # calculamos la piramide laplaciana de las imagenes
    # A y B
    lAs = compute_laplacian_pyramid(imgA)
    lBs = compute_laplacian_pyramid(imgB)
    
    lSs = []
    for lA, lB, GR, mA, mB in zip(lAs, lBs, gaussian_mask, masksA, masksB):
        # creamos la componente que albergará la
        # suma ponderada de las componentes de A y B
        new_component = np.zeros(lA.shape)
        
        # allá donde la mascara de A esté activa copiamos
        # la componente de A
        np.copyto(new_component, lA, where=mA > 0.5)
        # allá donde la mascara de B esté activa copiamos
        # la componente de B
        np.copyto(new_component, lB, where=mB > 0.5)

        # alla donde ambas componentes activas copiamos la
        # mezcla ponderada tal como especifica el algoritmo de
        # burt-adelson
        np.copyto(new_component, lA * GR + lB * (1 - GR), where=GR > 0)
        
        # añadimos la nueva componente a la lista de componentes
        lSs.append(new_component)

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
            cv2.addWeighted(
                imgIzq, 1,
                cv2.pyrUp(imgDer)[:height, :width], -1,
                0
            )
        )

    return pyramid

def compute_gaussian_pyramid(img, levels=4):
    pyramid = [img]
    
    downsampled = img
    for _ in range(levels):
        downsampled = cv2.pyrDown(downsampled)
        pyramid.append(downsampled)

    return pyramid
