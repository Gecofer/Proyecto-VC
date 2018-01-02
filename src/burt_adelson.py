import cv2
import numpy as np

from math import floor, sqrt

def burt_adelson(imgA, imgB, mask):
    gaussian_mask = compute_gaussian_pyramid(mask, levels=4)
    lAs = compute_laplacian_pyramid(imgA)
    lBs = compute_laplacian_pyramid(imgB)
    
    lSs = []
    for lA, lB, GR in zip(lAs, lBs, gaussian_mask):
        lSs.append(
            cv2.addWeighted(
                lA, GR,
                lB, 1 - GR,
                0
            )
        )

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
