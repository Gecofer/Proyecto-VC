import numpy as np

from math import floor, sqrt


def cylindrical_warp(img, f=20):
    """Performs a cylindrical warp on a given image"""

    # obtener las coordenadas de la proyección cilíndrica
    # libro Szeliski: formula 9.13 y 9.14
    # r=f gives less distortion
    def convert_coordinates(new_point, new_shape, f, r):

        # x' = r * arctang((x - x_centro) / f) + x_centro
        # y' = r * ((y - y_centro) / ((x - x_centro)**2 + f**2)) + y_centro

        y, x = (
            new_point[0] - new_shape[0]/2,
            new_point[1] - new_shape[1]/2
        )

        new_y = r * (y / sqrt(x**2 + f**2))
        new_x = r * np.arctan(x / f)

        return (
            floor(new_y) + new_shape[0]//2,
            floor(new_x) + new_shape[1]//2
        )

    # nos creamos un lienzo para pegar la
    # proyección cilindrica de la imagen
    new_img = np.zeros(img.shape, dtype=np.uint8)

    # recorremos la imagen de entrada (matriz)
    for row_index in range(len(img)):
        for col_index in range(len(img[0])):

            # convertimos las coordenadas de la
            # imagen
            y, x = convert_coordinates(
                (row_index, col_index),
                img.shape[:2],
                f,
                f
            )

            new_img[y, x] = img[row_index, col_index]

    return new_img

def spherical_warp(img, f=20):
    """Performs a spherical warp on a given image"""

    # obtener las coordenadas de la proyección esférica
    # libro Szeliski: formula 9.18 y 9.19
    # r=f gives less distortion
    def convert_coordinates(new_point, new_shape, f, r):

        # x' = r * arctang((x - x_centro) / f) + x_centro
        # y' = r * arctang( ((y - y_centro) / ((x - x_centro)**2 + f**2)) ) + y_centro

        y, x = (
            new_point[0] - new_shape[0]/2,
            new_point[1] - new_shape[1]/2
        )

        new_y = r * np.arctan( y / sqrt(x**2 + f**2))
        new_x = r * np.arctan(x / f)

        return (
            floor(new_y) + new_shape[0]//2,
            floor(new_x) + new_shape[1]//2
        )

    # nos creamos un lienzo para pegar la
    # proyección esférica de la imagen
    new_img = np.zeros(img.shape, dtype=np.uint8)

    # recorremos la imagen de entrada (matriz)
    for row_index in range(len(img)):
        for col_index in range(len(img[0])):

            # convertimos las coordenadas de la
            # imagen
            y, x = convert_coordinates(
                (row_index, col_index),
                img.shape[:2],
                f,
                f
            )

            new_img[y, x] = img[row_index, col_index]

    return new_img
