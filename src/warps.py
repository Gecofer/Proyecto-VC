import cv2
import numpy as np

from math import floor, sqrt

def cylindrical_warp(img, f=20):
    """Performs a cylindrical warp on a given image"""

    # la implementacion viene de
    # https://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection

    def convert_coordinates(new_point, new_shape, f, r):

        # r=f gives less distortion

        y, x = (
            new_point[0] - new_shape[0]/2,
            new_point[1] - new_shape[1]/2
        )

        new_y = f * ( y / sqrt(x**2 + f**2))
        new_x = f * np.arctan(x / f)

        return (
            floor(new_y) + new_shape[0]//2,
            floor(new_x) + new_shape[1]//2
        )

    # Obtenemos el ancho y el alto de la imagen
    height, width = img.shape[:2]

    new_img = np.zeros(img.shape, dtype=np.uint8)

    for i in range(len(img)):
        for j in range(len(img[0])):

            y, x = convert_coordinates(
                (i, j),
                img.shape[:2],
                f,
                f
            )

            new_img[y, x] = img[i, j]

    return new_img

