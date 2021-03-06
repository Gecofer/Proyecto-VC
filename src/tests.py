import cv2
import numpy as np
from matplotlib.pyplot import imread

from warps import cylindrical_warp, spherical_warp
from mosaic import mosaic
from util import show, plot_images, Image
from burt_adelson import compute_gaussian_pyramid, compute_laplacian_pyramid, collapse_laplacian_pyramid, burt_adelson


def test_warp():
    """Computes and displays a cylindrical and spherical warp over a white image"""
    img = np.ones((600, 600))
    mondrian = imread('../images/mondrian.jpg')

    show(cylindrical_warp(img, f=600))

    show(cylindrical_warp(mondrian, f=20))
    show(cylindrical_warp(mondrian, f=400))

    show(spherical_warp(img, f=600))

    show(spherical_warp(mondrian, f=20))
    show(spherical_warp(mondrian, f=400))

def test_guernica_cylindrical():
    """Computes and displays a cylindrical warp over a image guernica"""
    guernicas = [
        imread('../images/guernica{}.jpg'.format(str(i)))
        for i in range(1, 7)
    ]

    warped_guernicas = [
        cylindrical_warp(guernica, f=700)
        for guernica in guernicas
    ]

    plot_images([
        Image(img=w_guernica, title="guernica")
        for w_guernica in warped_guernicas
    ])

    mosaico = mosaic(warped_guernicas)

    show(mosaico)

def test_guernica_spherical():
    """Computes and displays a spherical warp over a image guernica"""
    guernicas = [
        imread('../images/guernica{}.jpg'.format(str(i)))
        for i in range(1, 7)
    ]

    warped_guernicas = [
        spherical_warp(guernica, f=1000)
        for guernica in guernicas
    ]

    plot_images([
        Image(img=w_guernica, title="guernica")
        for w_guernica in warped_guernicas
    ])

    mosaico = mosaic(warped_guernicas)

    show(mosaico)

def test_mosaic():
    """Computes and displays a mosaic"""
    guernicas = [
        imread('../images/guernica{}.jpg'.format(str(i)))
        for i in range(1, 4)
    ]

    show(mosaic(guernicas))

def test_mosaic_2():
    """Computes and displays a mosaic changing lighting """

    # funcion para cambiar la luminosidad de una imagen
    lum = lambda img, i: np.where(
        np.uint32(img) + i*20 < 255,
        img+ i*20,
        255
    )

    alhambras = [
        lum(imread("../images/alhambra{}.jpg".format(str(i))), i) //2
        for i in range(1, 6)
    ]
    
    show(*mosaic(alhambras)) # mostramos el bueno y el malo

def test_myselves():
    """Computes and displays mosaic with the algorith Burt and Adelson"""
    myselves = [
        imread("../images/myself/medium0{}.jpg".format(i))
        for i in range(1, 5)
    ]

    show(*mosaic(myselves))

def test_myselves_warped_and_lum():
    # funcion para cambiar la luminosidad de una imagen
    lum = lambda img, i: np.where(
        np.uint32(img) + i*20 < 255,
        img+ i*20,
        255
    )
    myselves = [
        cylindrical_warp(
            lum(imread("../images/myself/medium0{}.jpg".format(i)), i)//2,
            f=1200
        )
        for i in range(1, 5)
    ]
    show(*myselves)
    a = mosaic(myselves)

    show(a[0])
    show(a[1])

def test_zijing():
    # funcion para cambiar la luminosidad de una imagen
    myselves = [
        cylindrical_warp(
            imread("../images/zijing/medium0{}.jpg".format(i)),
            f=1000
        )
        for i in range(1, 10)
    ]
    show(*myselves)
    a = mosaic(myselves)

    show(a[0])
    show(a[1])



'''NO FUNCIONA ASÍ'''
def test_burt_adelson():
    """Computes and displays the algorith Burt and Adelson"""
    orange = cv2.imread("../images/orange.jpg", 1)
    orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)
    apple = cv2.imread("../images/apple.jpg", 1)
    apple = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
    mask = cv2.imread("../images/mask.jpg", 1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    show(burt_adelson(orange, apple, mask))

def test_gaussian_pyramid():
    """Computes and displays a gaussian pyramid"""
    guernica = imread('../images/guernica3.jpg')
    
    show(*compute_gaussian_pyramid(guernica))

'''NO FUNCIONA ASÍ'''
def test_laplacian_pyramid():
    """Computes and displays a laplacian pyramid"""
    guernica = imread('../images/guernica3.jpg')
    pyr = compute_laplacian_pyramid(guernica)
    reconstructed = collapse_laplacian_pyramid(pyr)
    show(*pyr, reconstructed)
