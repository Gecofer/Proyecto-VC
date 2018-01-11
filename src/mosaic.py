import cv2
import numpy as np

from burt_adelson import burt_adelson
from util import show

def mosaic(imgs):
    """Situa en un mosaico una lista de imagenes contiguas"""

    # en vez de superponer las imágenes se realiza una
    # mezcla con el algoritmo burt_adelson

    homographies = compute_homographies(imgs)

    # comenzamos la creacion del mosaico

    # el ancho depende del numero de imagenes que pongamos
    width = imgs[0].shape[1]*len(imgs)
    # el alto en principio no va a ser mucho mas grande que
    # el alto de una sola imagen
    height = imgs[0].shape[0] * 2

    size = height, width
    canvas = np.zeros(size + (3,), dtype=np.uint8)

    # el indice de la imagen que situaremos en el centro
    center = len(imgs) // 2
    center_img = imgs[center]
    center_height, center_width = center_img.shape[:2]

    # la traslacion al centro de la imagen
    tras = translation_matrix(
        width//2 - center_width//2,
        height//2 - center_height//2
    )

    # colocamos la imagen central
    canvas = perspective(center_img, tras, size)

    # pegamos las imagenes por la izquierda
    # empezamos por la inmediatamente a la izquierda
    # de la central y hasta la primera
    homography = tras
    for img_index in reversed(range(center)):
        
        # componemos la homografia que lleva de la
        # imagen i a la i+1 con la homografia acumulada
        homography = np.dot(
            homography,
            homographies[img_index]
        )

        tmp_canvas = perspective(
            imgs[img_index],
            homography,
            size
        )

        # usaremos la mascara para generar los pesos de cada imagen
        # para cada pixel de una nueva piŕamide Laplaciana
        mask = (canvas > 0).astype(np.uint8) * 255

        print('showing')
        show(burt_adelson(canvas, tmp_canvas, mask))

        canvas = burt_adelson(canvas, tmp_canvas, mask)

    homography = tras

    # pegamos las imagenes por la derecha
    # empezamos por la central y hasta la última
    for img_index in range(center + 1, len(imgs)):
        # se acumula la homografia. Hay que calcular
        # la inversa porque nuestra homografia lleva de
        # la i a la i + 1 pero queremos la que lleva de la
        # (i+1) a la i
        homography = np.dot(
            homography,
            np.linalg.inv(homographies[img_index - 1])
        )
        
        tmp_canvas = perspective(
            imgs[img_index],
            homography,
            size
        )

        mask = (canvas > 0).astype(np.float64)
        canvas = burt_adelson(canvas, tmp_canvas, mask)

    return canvas

def translation_matrix(x, y):
    """Devuelve la matriz de una traslacion"""
    return np.array([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ], dtype=float)

def perspective(src, M, size):
    """Realiza una homografia sobre una imagen de fondo negro"""
    return cv2.warpPerspective(
         src=src,
         M=M,
         dsize=size[::-1],
         dst=np.zeros(size, dtype=np.float32),
         borderMode=cv2.BORDER_CONSTANT
    )

def compute_homographies(imgs):
    """Calcula las homografias entre una lista de imagenes contiguas"""
    sift = cv2.xfeatures2d.SIFT_create()

    # calculamos las correspondencias
    matcher = cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=True
    )
    
    keypoints = []
    descriptors = []
    homographies = []

    # obtenemos todos los keypoints y descriptores
    # asociados de las imagenes
    for img in imgs:
        kps, desc = sift.detectAndCompute(img, mask=None)
        keypoints.append(kps)
        descriptors.append(desc)

    keypoints = np.array(keypoints)
    descriptors = np.array(descriptors)

    # calculamos las homografias entre cada par
    # de imagenes contiguas
    for i in range(len(imgs) - 1):
        matches = matcher.match(descriptors[i], descriptors[i+1])

        keypoints_izq = np.array([
            keypoints[i][match.queryIdx].pt
            for match in matches
        ])

        keypoints_der = np.array([
            keypoints[i+1][match.trainIdx].pt
            for match in matches
        ])

        homography, _ = cv2.findHomography(
            srcPoints=keypoints_izq,
            dstPoints=keypoints_der,
            method=cv2.RANSAC,
            ransacReprojThreshold=1
        )
        
        homographies.append(homography)

    return homographies
