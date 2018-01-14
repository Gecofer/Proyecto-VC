import cv2
import numpy as np

from matplotlib.pyplot import imread

from burt_adelson import burt_adelson
from util import show

def compute_largest_rectangle(mask):
    non_empty_cols = (mask > 0).any(axis=(0, 2))
    non_empty_rows = (mask > 0).any(axis=(1, 2))

    l = np.argwhere(non_empty_rows)
    first_row = l[0][0]
    last_row = l[-1][0]

    l = np.argwhere(non_empty_cols)
    first_col = l[0][0]
    last_col = l[-1][0]

    room = -10
    return first_row+room, last_row-room, first_col+room-20, last_col-room+20


def get_mask_from_corners(size, arriba, abajo):
    mask = np.uint8(
        cv2.fillConvexPoly(
            np.zeros(size),
            np.array(arriba + abajo[::-1]),
            color=255
        )
    )

    mask = np.array([mask, mask, mask]).transpose(1, 2, 0)

    return mask

def get_half_mask(first_row, last_row, first_col, last_col):
    white = np.ones((
        (last_row-first_row),
        (last_col-first_col)//2
    )) *255

    black = np.zeros((
        (last_row-first_row),
        (last_col-first_col) - (last_col-first_col)//2
    ))

    mask_to_mix = np.hstack(
        (black, white)
    ).astype(np.uint8)
    mask_to_mix = np.array([mask_to_mix]*3).transpose(1, 2, 0)

    return mask_to_mix

def float_image_to_uint8(img):
    return np.uint8(img* 255)

# Implementacion para la mascara: https://docs.opencv.org/3.0.0/d0/d86/tutorial_py_image_arithmetics.html

def mosaic(imgs):
    """Situa en un mosaico una lista de imagenes contiguas"""

    # en vez de superponer las imágenes se realiza una
    # mezcla con el algoritmo burt_adelson

    homographies = compute_homographies(imgs)

    # comenzamos la creacion del mosaico

    # el ancho depende del numero de imagenes que pongamos
    width = imgs[0].shape[1]*len(imgs)
    # el alto en principio no va a ser mucho mas grande que
    # el alto de una sola imagen
    height = imgs[0].shape[0] * 2

    size = height, width
    canvas = np.zeros(size + (3,), dtype=np.uint8)

    # el indice de la imagen que situaremos en el centro
    center = len(imgs) // 2
    center_img = imgs[center]
    center_size = center_img.shape[:2]
    center_height, center_width = center_size

    # la traslacion al centro de la imagen
    tras = translation_matrix(
        width//2 - center_width//2,
        height//2 - center_height//2
    )

    # colocamos la imagen central
    canvas = perspective(center_img, tras, size)
    # aqui llevaremos un canvas con colores mal mezclados
    # para compararlos luego y sacar la imagen en la memoria
    canvas2 = np.copy(canvas)
    
    # aqui iremos haciendo calculos temporales
    canvas_ = np.copy(canvas)
    canvas_[:, :, :] = 0

    # aqui llevaremos las esquinas de las imagenes para conocer
    # que region ocupan dentro del canvas
    arriba, abajo = compute_corner_coordinates(tras, center_size)

    # pegamos las imagenes por la izquierda
    # empezamos por la inmediatamente a la izquierda
    # de la central y hasta la primera
    homography = tras
    for img_index in reversed(range(center)):
        
        # componemos la homografia que lleva de la
        # imagen i a la i+1 con la homografia acumulada
        homography = np.dot(
            homography,
            homographies[img_index]
        )

        # la nueva imagen a pegar colocada en un marco negro
        tmp_canvas = perspective(
            imgs[img_index],
            homography,
            size
        )

        # obtenemos las esquinas de la nueva imagen
        current_img_corners_arriba, current_img_corners_abajo = \
            compute_corner_coordinates(
                homography,
                imgs[img_index].shape[:2]
            )

        # esta es la máscara que controla en qué parte del canvas está
        # el mosaico acumulado
        mask = get_mask_from_corners(size, arriba, abajo)
        
        # esta máscara controla la parte del canvas donde está
        # la nueva imagen colocada
        copy_img = imgs[img_index].copy()
        copy_img[:,:] = np.array([255, 255, 255])
        tmp_mask = perspective(copy_img, homography, size)
        # tmp_mask = get_mask_from_corners(
        #     size,
        #     current_img_corners_arriba,
        #     current_img_corners_abajo
        # )

        # sacamos el mínimo de ambas máscaras, lo que mide
        # por tanto la zona de intersección de ambas imágenes
        shared_mask = np.where(
            mask < tmp_mask,
            mask,
            tmp_mask
        )

        # obtenemos las esquinas de un rectángulo horizontal
        # contenido dentro de la intersección. Es en este
        # rectángulo donde haremos Burt-Adelson
        first_row, last_row, first_col, last_col = \
            compute_largest_rectangle(shared_mask)

        # subimagen del mosaico acumulado donde haremos
        # burt-adelson
        canvas_to_mix = canvas[
            first_row:last_row,
            first_col:last_col
        ]

        # subimagen del nuevo canvas donde haremos burt-adelson
        tmp_canvas_to_mix = tmp_canvas[
            first_row:last_row,
            first_col:last_col
        ]

        # en el canvas resultado podemos pegar todo lo que tenemos
        # fuera de la zona donde vamos a hacer burt-adelson
        canvas_[:, :first_col+3] = tmp_canvas[:, :first_col+3]
        canvas_[:, last_col-3:] = canvas[:, last_col-3:]


        # obtenemos una máscara que divide por la mitad
        # la subimagen en la que vamos a hacer burt-adelson
        mask_to_mix = get_half_mask(
            first_row,
            last_row,
            first_col,
            last_col
        )

        import pdb; pdb.set_trace()
        # hacemos burt-adelson en la subimagen determinada
        roi = float_image_to_uint8(
            burt_adelson(
                canvas_to_mix,
                tmp_canvas_to_mix,
                mask_to_mix
            )
        )

        
        canvas2 = np.where(canvas2 > tmp_canvas, canvas2, tmp_canvas)

        # actualizamos el canvas con la zona burt-adelson
        canvas_[first_row:last_row, first_col:last_col] = roi

        canvas = canvas_

        # añadimos las esquinas de la última imagen a la lista
        arriba = [current_img_corners_arriba[0]] + arriba
        abajo = [current_img_corners_abajo[0]] + abajo

    homography = tras

    # pegamos las imagenes por la derecha
    # empezamos por la central y hasta la última
    for img_index in range(center + 1, len(imgs)):
        # se acumula la homografia. Hay que calcular
        # la inversa porque nuestra homografia lleva de
        # la i a la i + 1 pero queremos la que lleva de la
        # (i+1) a la i
        homography = np.dot(
            homography,
            np.linalg.inv(homographies[img_index - 1])
        )
        
        # la nueva imagen a colocar colocada en un marco negro
        tmp_canvas = perspective(
            imgs[img_index],
            homography,
            size
        )

        # obtenemos las coordenadas
        # de las esquinas de la nueva imagen que vamos
        # a pegar
        current_img_corners_arriba, current_img_corners_abajo = \
            compute_corner_coordinates(
                homography,
                imgs[img_index].shape[:2]
            )

        # esta mascara controla donde esta el mosaico
        # acumulado
        mask = get_mask_from_corners(size, arriba, abajo)
        
        # mascara que controla donde esta la nueva imagen
        # dentro del canvas
        copy_img = imgs[img_index].copy()
        copy_img[:,:] = np.array([255, 255, 255])
        tmp_mask = perspective(copy_img, homography, size)

        # tmp_mask = get_mask_from_corners(
        #     size,
        #     current_img_corners_arriba,
        #     current_img_corners_abajo
        # )

        

        # obtenemos el minimo de ambas mascaras
        # esto es equivalente a obtener la interseccion de
        # ambas imagenes
        shared_mask = np.where(
            mask < tmp_mask,
            mask,
            tmp_mask
        )

        # obtenemos coordenadas
        # del mayor rectangulo que podemos introducir
        # dentro de esa mascara. Es donde haremos
        # burt-adelson
        first_row, last_row, first_col, last_col = \
            compute_largest_rectangle(shared_mask)

        # subimagen del canvas acumulado donde haremos
        # burt-adelson
        canvas_to_mix = canvas[
            first_row:last_row,
            first_col:last_col
        ]

        # subimagen del nuevo canvas donde haremos
        # burt-adelson
        tmp_canvas_to_mix = tmp_canvas[
            first_row:last_row,
            first_col:last_col
        ]

        # mascara que lleva la mitad del rectangulo
        # en 1s y la otra en 0. Esta invertida respecto
        # a la del anterior bucle porque aqui pegamos
        # las nuevas imagenes a la derecha
        mask_to_mix = 255 - get_half_mask(
            first_row,
            last_row,
            first_col,
            last_col
        )

        # en el canvas resultado podemos pegar todo lo que tenemos
        # fuera de la zona donde vamos a hacer burt-adelson
        canvas_[:, :first_col+3] = canvas[:, :first_col+3]
        canvas_[:, last_col-3:] = tmp_canvas[:, last_col-3:]

        # hacemos burt-adelson en la subimagen
        # de la que hablabamos
        roi = float_image_to_uint8(
            burt_adelson(
                canvas_to_mix,
                tmp_canvas_to_mix,
                mask_to_mix
            )
        )

        canvas2 = np.where(canvas2 > tmp_canvas, canvas2, tmp_canvas)

        # ponemos la region de burt-adelson en nuestro canvas
        canvas_[first_row:last_row, first_col:last_col] = roi



        canvas = canvas_

        arriba = arriba + [current_img_corners_arriba[1]] 
        abajo =  abajo + [current_img_corners_abajo[1]]


    # devolvemos un canvas con los colores mal mezclados
    # y otro con los colores bien mezclados para ver la diferencia
    return canvas, canvas2

def compute_corner_coordinates(homography, size):
    height, width = size

    apply = lambda p: homography.dot(p + [1]).astype(int)[:2]
    room = 1
    
    top_left = apply([room, room])
    top_right = apply([width - room, room])
    bottom_left = apply([room, height-room])
    bottom_right = apply([width - room, height-room]) # deja un par
    # de pixeles de sobra para no coger el borde

    return [top_left, top_right], [bottom_left, bottom_right]

def translation_matrix(x, y):
    """Devuelve la matriz de una traslacion"""
    return np.array([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ], dtype=float)

def perspective(src, M, size):
    b = np.zeros(size)
    """Realiza una homografia sobre una imagen de fondo negro"""
    return cv2.warpPerspective(
        src=src,
        M=M,
        dst=b,
        dsize=size[::-1],
        borderMode=cv2.BORDER_CONSTANT
    )

def compute_homographies(imgs):
    """Calcula las homografias entre una lista de imagenes contiguas"""
    sift = cv2.xfeatures2d.SIFT_create()

    # calculamos las correspondencias
    matcher = cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=True
    )
    
    keypoints = []
    descriptors = []
    homographies = []

    # obtenemos todos los keypoints y descriptores
    # asociados de las imagenes
    for img in imgs:
        kps, desc = sift.detectAndCompute(img, mask=None)
        keypoints.append(kps)
        descriptors.append(desc)

    keypoints = np.array(keypoints)
    descriptors = np.array(descriptors)

    # calculamos las homografias entre cada par
    # de imagenes contiguas
    for i in range(len(imgs) - 1):
        matches = matcher.match(descriptors[i], descriptors[i+1])

        keypoints_izq = np.array([
            keypoints[i][match.queryIdx].pt
            for match in matches
        ])

        keypoints_der = np.array([
            keypoints[i+1][match.trainIdx].pt
            for match in matches
        ])

        homography, _ = cv2.findHomography(
            srcPoints=keypoints_izq,
            dstPoints=keypoints_der,
            method=cv2.RANSAC,
            ransacReprojThreshold=1
        )
        
        homographies.append(homography)

    return homographies


# guernicas = [
#     imread('../images/guernica{}.jpg'.format(str(i)))
#     for i in range(1, 3)
# ]

# show(mosaic(guernicas))
