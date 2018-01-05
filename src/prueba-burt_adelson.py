import cv2
import numpy as np

from matplotlib.pyplot import imread
from math import floor, sqrt
from util import show, plot_images, Image


# Implementación: https://github.com/yrevar/semi_automated_cinemagraph/blob/main/blending_utils.py

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

def equation_blend(laplacianA, laplacianB, gaussianMask):

    blend = []

    for lS in range(len(laplacianB)):
        blend.append(gaussianMask[lS]*laplacianA[lS] + (1-gaussianMask[lS])*laplacianB[lS])

    return blend


def generating_kernel(a):

    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)

def expand_l(image, kernel=generating_kernel(0.4)):

    H,W = image.shape
    # create output image
    out_img = np.zeros((2*H, 2*W), dtype=np.float64)
    out_img[::2,::2] = image
    # convolve
    out_img = 4*cv2.filter2D(out_img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return out_img

def collapse(pyramid):

    prev_lvl_img = pyramid[-1]

    for curr_lvl in range(len(pyramid)-1)[::-1]:

        prev_lvl_img_expand = expand_l(prev_lvl_img)

        if pyramid[curr_lvl].shape != prev_lvl_img_expand.shape:
            prev_lvl_img = pyramid[curr_lvl] +\
                        prev_lvl_img_expand[:pyramid[curr_lvl].shape[0],:pyramid[curr_lvl].shape[1]]

        else:
            prev_lvl_img = pyramid[curr_lvl] + prev_lvl_img_expand

    return prev_lvl_img

def burt_adelson(imgA, imgB, mask):

    gaussian_mask = compute_gaussian_pyramid(mask, levels=4)
    maskA = compute_gaussian_pyramid(imgA, levels=4)
    maskB = compute_gaussian_pyramid(imgB, levels=4)

    lA = compute_laplacian_pyramid(imgA, levels=4)
    lB = compute_laplacian_pyramid(imgB, levels=4)

    pyramid = equation_blend(lA, lB, gaussian_mask)
    img_blend = collapse(pyramid)

    return [img_blend]
   # return (maskA, maskB, gaussian_mask,
    #        lA, lB, pyramid, [img_blend])

def visualize_pyr(pyramid):

    """Create a single image by vertically stacking the levels of a pyramid."""
    shape = np.atleast_3d(pyramid[0]).shape[:-1]  # need num rows & cols only
    img_stack = [cv2.resize(layer, shape[::-1],
                            interpolation=3) for layer in pyramid]
    return np.vstack(img_stack).astype(np.uint8)


def blend_and_store_images(black_image, white_image, mask):

    """Apply pyramid blending to each color channel of the input images """

    # Convert to double and normalize the images to the range [0..1]
    # to avoid arithmetic overflow issues
    b_img = np.atleast_3d(black_image).astype(np.float) / 255.
    w_img = np.atleast_3d(white_image).astype(np.float) / 255.
    m_img = np.atleast_3d(mask).astype(np.float) / 255.
    num_channels = b_img.shape[-1]

    imgs = []
    for channel in range(num_channels):
        imgs.append(burt_adelson(b_img[:, :, channel],
                              w_img[:, :, channel],
                              m_img[:, :, channel]))

    names = ['outimg']

    for name, img_stack in zip(names, zip(*imgs)):
        imgs = map(np.dstack, zip(*img_stack))
        stack = [cv2.normalize(img, dst=None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX)
                 for img in imgs]

        cv2.imwrite('../images/outimg.png', visualize_pyr(stack))

orange = cv2.imread("../images/orange.jpg",1)
apple = cv2.imread("../images/apple.jpg",1)
mask = cv2.imread("../images/mask.jpg",1)



(blend_and_store_images(orange, apple, mask))

