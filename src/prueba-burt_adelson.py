import cv2
import numpy as np

from math import *

# Implementación: https://github.com/yrevar/semi_automated_cinemagraph/blob/main/blending_utils.py
# Implementación: https://github.com/stheakanath/multiresolutionblend/blob/master/main.py

# Usamos nuestra función Gaussiana
# --------------------------------------------------------------------------------
def compute_gaussian_pyramid(img, levels=4):
	pyramid = [img]

	downsampled = img
	for _ in range(levels):
		downsampled = cv2.pyrDown(downsampled)
		pyramid.append(downsampled)

	return pyramid


# Para la función Laplaciana, usamos https://github.com/yrevar/semi_automated_cinemagraph/blob/main/blending_utils.py
# --------------------------------------------------------------------------------
def generating_kernel(a):

	kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
	return np.outer(kernel, kernel)

def expand_l(image, kernel=generating_kernel(0.4)):

	H, W = image.shape
	# create output image
	out_img = np.zeros((2*H, 2*W), dtype=np.float64)
	out_img[::2,::2] = image
	# convolve
	out_img = 4*cv2.filter2D(out_img, -1, kernel, borderType=cv2.BORDER_REFLECT)
	return out_img

def lapl_pyr(gaussPyr):

	# level 0 is same is the top level of gaussPyr
	l_pyr = [gaussPyr[-1]]

	# iterate in reverse from (top level - 1) to 0
	for i in range(len(gaussPyr)-1)[::-1]:

		# exapand the image from the level above current
		expand_image = expand_l(gaussPyr[i+1])
		# current level image
		g_pyr_img = gaussPyr[i]

		# check if these two images are aligned
		if g_pyr_img.shape != expand_image.shape:

			# NOTE: if misaligned then crop the residual rows and columns before taking
			# difference
			l_pyr.append(g_pyr_img-expand_image[:g_pyr_img.shape[0],:g_pyr_img.shape[1]])
		else:
			# compute difference: laplacian image at current scale
			l_pyr.append(g_pyr_img-expand_image)

	return l_pyr[::-1]


# Para calcular la ecuación de los pesos de Burt Adelson
# --------------------------------------------------------------------------------
def equation_blend(laplacianA, laplacianB, gaussianMask):
	blend = []

	for lS in range(len(laplacianB)):
		blend.append(gaussianMask[lS]*laplacianA[lS] + (1-gaussianMask[lS])*laplacianB[lS])

	return blend


# Para reconstruir la pirámide https://github.com/yrevar/semi_automated_cinemagraph/blob/main/blending_utils.py
# --------------------------------------------------------------------------------
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


# Para crear el algoritmo Burt-Adelson
# --------------------------------------------------------------------------------
def burt_adelson(imgA, imgB, mask):

	gaussian_mask = compute_gaussian_pyramid(mask, levels=4)
	maskA = compute_gaussian_pyramid(imgA, levels=4)
	maskB = compute_gaussian_pyramid(imgB, levels=4)

	lA = lapl_pyr(maskA)
	lB = lapl_pyr(maskB)

	pyramid = equation_blend(lA, lB, gaussian_mask)
	img_blend = collapse(pyramid)

	return [img_blend]
	# return (maskA, maskB, gaussian_mask, lA, lB, pyramid, [img_blend])


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

#show(*compute_gaussian_pyramid(mask))
#show(*compute_laplacian_pyramid(apple))
#show(piramide_laplaciana(apple, levels=5))
#h = (lapl_pyr(compute_gaussian_pyramid(orange)))
#show(*h)
(blend_and_store_images(orange, apple, mask))

