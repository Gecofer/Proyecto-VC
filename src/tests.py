import numpy as np
import numpy as np
from matplotlib.pyplot import imread

from warps import cylindrical_warp
from mosaic import mosaic
from util import show, plot_images, Image


def test_warp():
    """Computes and displays a cylindrical warp over a white image"""
    img = np.ones((600, 600))
    mondrian = imread('../images/mondrian.jpg')

    show(cylindrical_warp(img, f=600))
    show(cylindrical_warp(mondrian, f=20))
    show(cylindrical_warp(mondrian, f=400))

def test_guernica():
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

def test_mosaic():
    guernicas = [
        imread('../images/guernica{}.jpg'.format(str(i)))
        for i in range(1, 7)
    ]

    show(mosaic(guernicas))
