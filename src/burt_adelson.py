import cv2
import numpy as np


# Implementación para las pirámides: https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html

normalize = lambda i: cv2.normalize(
    i,
    dst=None,
    alpha=0,
    beta=1,
    norm_type=cv2.NORM_MINMAX
)

def compute_gaussian(img, levels=7):
    """Realiza una pirámide gaussiano en una imagen dada"""

    # guardamos el primer nivel de la pirámide
    pyramid = [img]
    acc = img

    for _ in range(levels):
        # redimensionar y alisar
        acc = cv2.pyrDown(acc)
        pyramid.append(acc)

    return pyramid

compute_gaussian_pyramid = compute_gaussian

def compute_laplacian(img, levels=7):
    """Realiza una pirámide laplaciana en una imagen dada"""

    laplacian = compute_gaussian(img, levels)
    pyramid = [laplacian[levels-1]]

    for i in range(levels-1,0,-1):
        GE = cv2.pyrUp(laplacian[i])
        height, width = laplacian[i-1].shape
        L = cv2.subtract(laplacian[i-1],GE[:height, :width])
        pyramid.append(L)

    return pyramid[::-1]

compute_laplacian_pyramid = compute_laplacian

def spline(pyramid):
    '''Recomponer la imagen colapsando la piramide'''
    prev = pyramid[-1]

    for img in reversed(pyramid[:-1]):
        height, width = img.shape[:2]
        prev = cv2.pyrUp(prev)
        prev = img + prev[:height, :width]

    return prev

collapse_laplacian_pyramid = spline

def blend(imgA, imgB, mask):
    ''' Forma una pirámide combinada con A y B,
    usando los nodos de la máscara como pesos'''

    levels = 6
    gpMask = compute_gaussian(mask, levels)

    lAs = compute_laplacian(imgA, levels)
    lBs = compute_laplacian(imgB, levels)

    lSs = []

    # LS_k(i,j) = GM_k(I,j,)*LA_k(I,j) + (1-GM_k(I,j))*LB_k(I,j)
    for lA, lB, G in zip(lAs, lBs, gpMask):
        lSs.append(G*lA + (1-G)*lB)

    return spline(lSs)

n_to_float = lambda img: img.astype(np.float)/255

def burt_adelson(imgA, imgB, mask):
    imgA = n_to_float(imgA)
    imgB = n_to_float(imgB)
    mask = n_to_float(mask)

    blended_channels = [
        blend(
            imgA[:, :, channel],
            imgB[:, :, channel],
            mask[:,:, channel]
        )
        for channel in range(len(imgA[0,0]))
    ]


    return np.array(blended_channels).transpose(1, 2, 0)
