import cv2
import numpy as np

from math import floor, sqrt
from util import show
from matplotlib.pyplot import imread


# Implementación: https://github.com/stheakanath/multiresolutionblend/blob/master/main.py
# Implementación: https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html

def compute_gaussian(img, levels=6):
    pyramid = []
    acc = img

    for _ in range(levels):
        acc = cv2.pyrDown(acc)
        pyramid.append(acc)

    return pyramid

compute_gaussian_pyramid = compute_gaussian

def compute_laplacian1(img, levels=6):
    g_pyramid = compute_gaussian(img)
    pyramid = []

    for imgIzq, imgDer in zip(g_pyramid, g_pyramid[1:]):
        pyramid.append(imgIzq - cv2.pyrUp(imgDer))
        
    return pyramid

# laplacian de tutorial OpenCV
def compute_laplacian(img, levels=6):
    laplacian = compute_gaussian(img, levels)
    pyramid = [laplacian[5]]

    for i in range(5,0,-1):
        GE = cv2.pyrUp(laplacian[i])
        L = cv2.subtract(laplacian[i-1],GE)
        pyramid.append(L)

    return pyramid[::-1]

compute_laplacian_pyramid = compute_laplacian

def blend_pipeline(imgA, imgB, mask):
    gpA = compute_gaussian(imgA)
    gpB = compute_gaussian(imgB)
    gpMask = compute_gaussian(mask)

    lAs = compute_laplacian(imgA)
    lBs = compute_laplacian(imgB)

    lSs = []

    for lA, lB, G in zip(lAs, lBs, gpMask):
        lSs.append(G * lA + (1-G) * lB)

    return collapse(lSs)

def collapse(pyramid):
    prev = pyramid[-1]

    for img in reversed(pyramid[:-1]):
        height, width = img.shape[:2]
        prev = cv2.pyrUp(prev)
        prev = img + prev[:height, :width]

    return prev

collapse_laplacian_pyramid = collapse

def burt_adelson(imgA, imgB, mask):
    # Convert to double and normalize the images to the range [0..1]
    # to avoid arithmetic overflow issues
    imgA = np.atleast_3d(imgA).astype(np.float) / 255.
    imgB = np.atleast_3d(imgB).astype(np.float) / 255.
    mask = np.atleast_3d(mask).astype(np.float) / 255.
    num_channels = imgB.shape[-1]

    imgs = []

    for channel in range(num_channels):
        v = blend_pipeline(
            imgA[:, :, channel],
            imgB[:, :, channel],
            mask[:, :, channel]
        )
        # los tres primero salen bien, pero el ultimo no
        # show(v)
        imgs.append(v)

    imgs = zip(*imgs)
    imgs = np.dstack(imgs).transpose(2, 1, 0)

    # si normalizas sale mal
    '''
    return cv2.normalize(
        imgs,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )
    '''
    return imgs


# para leer en blanco y negro pongo un 0
# y para leer en color pongo un 1 y
# la linea de cv2.cvtColor
orange = cv2.imread("../images/orange.jpg", 1)
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)
apple = cv2.imread("../images/apple.jpg", 1)
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
mask = cv2.imread("../images/mask.jpg", 1)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# con blanco y negro sale bien
# show(burt_adelson(orange, apple, mask))

# con esto obtengo la salida que ayer viste
# se diferencia algo con colores saturados
#show(blend_pipeline(orange, apple, mask))

# con esto la salida es mala
# comprobar último nivel de las pirámides
# show(burt_adelson(orange, apple, mask))


# CON ESTO LA SALIDA ES BUENA Y USO
# (compute_laplacian) DE OPENCV TUTORIAL
show(burt_adelson(orange, apple, mask))




