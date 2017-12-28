import cv2
import numpy as np
from matplotlib.pyplot import imread

from math import floor, tan, sqrt, floor
from util import plot_images, Image, show

def cylindrical_warp(img, f=20):
    """Performs a cylindrical warp on a given image"""

    # la implementacion viene de
    # https://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection

    def convert_coordinates(new_point, new_shape, f, r):

        y, x = (
            new_point[0] - new_shape[0]//2,
            new_point[1] - new_shape[1]//2
        )

        new_y = y * sqrt(1 + tan(x / f) ** 2)
        new_x = f * tan(x / f)

        return (
            floor(new_y) + new_shape[0]//2,
            floor(new_x) + new_shape[1]//2
        )

    height, width = img.shape[:2]
    print(height,width)
    new_img = np.zeros(img.shape, dtype=np.uint8)

    for row_index in range(len(img)):
        for col_index in range(len(img[0])):
            y, x = convert_coordinates(
                (row_index, col_index),
                img.shape[:2],
                f,
                f
            )

            if 0 <= x < width and 0 < y < height:
                new_img[row_index, col_index] = img[y, x]

    return new_img


# Función que genera un Mosaico para N imágenes relacionadas
def mosaicoN(lista_imagenes):

    '''
    :param lista_imagenes: imágenes ordenadas de izquierda a derecha para generar el mosaico

    :return: mosaico de salida imprimido en un imagen con fondo negro.
    '''

    # Nos creamos las variables necesarias a usar
    coincidencias = []
    keypoints = []
    homografia = []
    descriptors = []

    # Clase para la coincidencia de descripciones keypoints
    # (query descriptor index, train descriptor index, train image index, and distance between descriptors)
    coincidencias.append(cv2.DMatch)

    # Para extraer los puntos clave y los descriptores utilizamos SIFT
    # Debemos tener acceso a la implementación original de SIFT, que están en el submódulo xfeatures2d
    sift = cv2.xfeatures2d.SIFT_create()

    # Para establecer las correspondencias entre las imágenes usaremos: Brute-force matcher create method
    correspondencias = cv2.BFMatcher( normType=cv2.NORM_L2, crossCheck=True)

    # Saco los keypoints y descriptores de cada imagen del mosaico
    for i in range(len(lista_imagenes)):

        # Detectar y extraer características de la imagen
        kps, desc = sift.detectAndCompute(image=lista_imagenes[i], mask=None)

        # Convertir los keypoints con estructura Keypoint a un Array
        kps = np.float32([kp.pt for kp in kps])

        # Guarda en un lista las características de cada imagen
        keypoints.append(kps)
        descriptors.append(desc)

    # Obtengo el vector de correspondencias y la homografía de cada par de imagenes adyacentes en el mosaico horizontal
    for i in range (len(lista_imagenes)-1):

        # Obtengo las correspondencias de la imagen i con la i+1
        matches = correspondencias.match(descriptors[i], descriptors[i+1])
        # Ordeno las coincicencias por el orden de la distancia
        matches = sorted(matches, key=lambda x: x.distance)
        coincidencias.append(matches[i])

        # Extraigo los keypoints de la imagen i que están en correspondencia con los keypoints de la imagen i+1
        keypoints_imagen1 = np.float32([keypoints[i][j.queryIdx] for j in matches])
        keypoints_imagen2 = np.float32([keypoints[i+1][j.trainIdx] for j in matches])

        # Calcular la homografía entre los dos conjuntos de puntos
        h, status = (cv2.findHomography(srcPoints=keypoints_imagen1, dstPoints=keypoints_imagen2, method=cv2.RANSAC, ransacReprojThreshold=1))
        homografia.append(h)

        # Borramos el contenido de dichos keypoints para usarlos la siguiente iteracion
        np.array([row for row in keypoints_imagen1 if len(row)<=3])
        np.array([row for row in keypoints_imagen2 if len(row)<=3])

    # Nos creamos un fondo negro, con un tamaño específico (para que quepan las demás fotografías)
    ancho = lista_imagenes[0].shape[0]*4
    alto = lista_imagenes[0].shape[1]*3

    # Obtenemos la imagen del centro
    centro = floor(len(lista_imagenes) / 2)

    # Definimos la traslacion que nos pone la imagen central del mosaico en el centro
    tras = np.matrix([[1, 0, lista_imagenes[centro].shape[1]], [0, 1, lista_imagenes[centro].shape[0]], [0, 0, 1]], dtype=np.float32)
    # Llevamos esa imagen al centro de nuestro mosaico con la homografia
    mosaico = cv2.warpPerspective(src=lista_imagenes[centro], M=tras, dsize=(ancho, alto), borderMode=cv2.BORDER_TRANSPARENT)

    # Calculamos las homografias que se le aplican a las imagenes de la izquierda de la imagen central
    for i in range(0, centro):

        # Definimos la traslacion para las imágenes de la izquierda
        izquierda = np.matrix([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.float32)
        #izquierda = np.eye(3, dtype=np.float32)

        for j in range(i, centro): izquierda = homografia[j] * izquierda

        # Las llevamos al mosaico
        cv2.warpPerspective(src=lista_imagenes[i], M=tras*izquierda, dst=mosaico, dsize=(ancho, alto), borderMode=cv2.BORDER_TRANSPARENT)

    # Calculamos las homografias que se le aplican a las imagenes de la derecha de la imagen central
    # Ahora debemos usar las inversas de las homografías, ya que las homografias que se usan son de la imagen i a la i-1
    for i in range(centro + 1, len(lista_imagenes)):

        # Definimos la traslacion para las imágenes de la derecha
        derecha = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        #derecha = np.eye(3, dtype=np.float32)

        for j in range(centro, i): derecha = derecha * np.linalg.inv(homografia[j])

        # Las llevamos al mosaico
        cv2.warpPerspective(lista_imagenes[i], M=tras*derecha, dst=mosaico, dsize=(ancho, alto),borderMode=cv2.BORDER_TRANSPARENT)

    return mosaico


def test_warp():
    """Computes and displays a cylindrical warp over a white image"""
    img = np.ones((600, 600))
    mondrian = imread('../images/mondrian.jpg')
    guernica1 = imread('../images/guernica1.jpg')
    guernica2 = imread('../images/guernica2.jpg')
    guernica3 = imread('../images/guernica3.jpg')
    guernica4 = imread('../images/guernica4.jpg')
    mosaico002 = imread('../images/mosaico002.jpg')
    mosaico003 = imread('../images/mosaico003.jpg')
    mosaico004 = imread('../images/mosaico004.jpg')
    mosaico005 = imread('../images/mosaico005.jpg')
    mosaico006 = imread('../images/mosaico006.jpg')

    show(cylindrical_warp(img, f=600))
    show(cylindrical_warp(img, f=50))
    #show(cylindrical_warp(mondrian, f=100))

    imageA = cylindrical_warp(mosaico002, f=200)
    imageB = cylindrical_warp(mosaico003, f=200)
    imageC = cylindrical_warp(mosaico004, f=200)
    imageD = cylindrical_warp(mosaico005, f=200)
    imageE = cylindrical_warp(mosaico006, f=200)

    mosaico = mosaicoN([imageA, imageB, imageC, imageD, imageE])

    #mosaico = mosaicoN([mosaico002, mosaico003, mosaico004])
    show(mosaico)
