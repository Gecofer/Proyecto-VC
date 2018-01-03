import cv2
import numpy as np

from math import floor, sqrt

def burt_adelson(imgA, imgB, maskA, maskB):
    shared_mask = (maskA & maskB).astype(float)
    maskA = maskA.astype(float)
    maskB = maskB.astype(float)

    gaussian_mask = compute_gaussian_pyramid(shared_mask, levels=4)
    masksA = compute_gaussian_pyramid(maskA, levels=4)
    masksB = compute_gaussian_pyramid(maskB, levels=4)

    lAs = compute_laplacian_pyramid(imgA)
    lBs = compute_laplacian_pyramid(imgB)
    
    lSs = []
    for lA, lB, GR, mA, mB in zip(lAs, lBs, gaussian_mask, masksA, masksB):
        new_component = np.zeros(lA.shape)
        
        new_component = np.where(
            mA > 0.5,
            lA,
            new_component
        )

        new_component = np.where(
            mB > 0.5,
            lB,
            new_component
        )

        new_component = np.where(
            GR,
            lA * GR + lB * (1 - GR),
            new_component
        )
        
        lSs.append(new_component)

    return lSs

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
