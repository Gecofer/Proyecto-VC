import matplotlib.pyplot as plt

from math import ceil
from collections import namedtuple
import numpy as np
import cv2

Image = namedtuple('Image', ['img', 'title'])

def plot_images(imgs):
    n_images = len(imgs)

    # encontramos el grid /óptimo/ para mostrar
    # las imaǵenes
    cols = ceil(n_images**(1/2))
    rows = ceil(n_images / cols)

    for i, image in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image.img, cmap='gray')
        plt.title(image.title)

    plt.tight_layout()
    plt.show()

show = lambda img: plot_images([Image(img=img, title='Image')])

