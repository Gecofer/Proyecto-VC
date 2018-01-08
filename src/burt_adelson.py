import cv2
import numpy as np

from math import floor, sqrt
from util import show
from matplotlib.pyplot import imread


# Implementación: https://github.com/stheakanath/multiresolutionblend/blob/master/main.py
# Implementación: https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html


def compute_gaussian_pyramid(img, levels=6):
    downsampled = img.copy()
    pyramid = [downsampled]

    for i in range(levels):
        downsampled = cv2.pyrDown(downsampled)
        pyramid.append(downsampled)

    return pyramid


def compute_laplacian_pyramid(img, levels=6):

    laplacian = compute_gaussian_pyramid(img, levels)
    pyramid = [laplacian[-1]]

    for i in range(5,0,-1):
    #for i in range(len(laplacian)-1)[::-1]:
        GE = cv2.pyrUp(laplacian[i])
        L = cv2.subtract(laplacian[i-1],GE)
        pyramid.append(L)

    return pyramid

def collapse_laplacian_pyramid(pyramid):


    result = np.zeros(pyramid[-1].shape)

    for img in reversed(pyramid):
        height, width = img.shape[:2]

        result = cv2.addWeighted(img, 1, result[:height, :width], 1, 0)
        result = cv2.pyrUp(result)

    return result
    '''
    lS = pyramid[0]

    for i in range(1,6):
        lS = cv2.pyrUp(lS)
        lS = cv2.add(lS, pyramid[i])

    return lS
    '''

def combine_laplacian_pyramids(laplacianA, laplacianB, gaussianMask):
    blend = []

    for lS in range(len(laplacianB)):
        blend.append(gaussianMask[lS]*laplacianA[lS] + (1-gaussianMask[lS])*laplacianB[lS])

    return blend


def burt_adelson(imgA, imgB, mask):

    # Step la. Build Laplacian pyramids LA and LB
    # for images A and B respectively.
    lAs = compute_laplacian_pyramid(imgA)
    lBs = compute_laplacian_pyramid(imgB)

    # Step lb. Build a Gaussian pyramid GR for the
    # region image R.
    gaussian_mask = compute_gaussian_pyramid(mask,
                                             levels=4)

    # Step 2. Form a combined pyramid LS from LA and LB
    # using nodes of GR as weights. That is, for each l, i and j:
    pyramid = combine_laplacian_pyramids(lAs, lBs, gaussian_mask)
    '''
    lSs = []
    for lA, lB, GR in zip(lAs, lBs, gaussian_mask):
        r = lA * GR + lB * (1 - GR)
        show(lA, lB, GR, lA * GR, lB * (1 - GR), r)
        lSs.append(r)
    '''
    # aqui habria que recomponer las componentes de la laplaciana
    img_blend = collapse_laplacian_pyramid(pyramid)

    return img_blend


def test_laplacian_pyramid():
    """Computes and displays a laplacian pyramid"""
    guernica = imread('../images/guernica3.jpg')
    pyr = compute_laplacian_pyramid(guernica)
    reconstructed = collapse_laplacian_pyramid(pyr)
    show(*pyr, reconstructed)

if __name__ == "__main__":
    test_laplacian_pyramid()

