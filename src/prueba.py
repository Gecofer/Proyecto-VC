import cv2
import numpy as np,sys
from util import show


def compute_gaussian_pyramid(img, levels=6):
	downsampled = img.copy()
	pyramid = [downsampled]

	for i in range(levels):
		downsampled = cv2.pyrDown(downsampled)
		pyramid.append(downsampled)

	return pyramid

def compute_laplacian_pyramid(img, levels=6):
	laplacian = compute_gaussian_pyramid(img, levels)
	pyramid = [laplacian[5]]

	for i in range(5,0,-1):
		GE = cv2.pyrUp(laplacian[i])
		L = cv2.subtract(laplacian[i-1],GE)
		pyramid.append(L)

	return pyramid[::-1]

def combine_laplacian_pyramids(laplacianA, laplacianB, gaussianMask):
	blend = []

	for lS in range(len(laplacianB)):
		blend.append(gaussianMask[lS]*laplacianA[lS] + (1-gaussianMask[lS])*laplacianB[lS])

	return blend


def expand_laplacian(image):

	# np.outer: Compute the outer product of two vectors.
	kernel = np.outer(cv2.getGaussianKernel(7,1), cv2.getGaussianKernel(7,1))

	H, W = image.shape

	# create output image
	out_img = np.zeros((2*H, 2*W), dtype=np.float64)
	out_img[::2,::2] = image

	# convolve
	out_img = 4 * cv2.filter2D(out_img, -1, kernel, borderType=cv2.BORDER_REFLECT)

	return out_img



def collapse_laplacian_pyramid(pyramid):
	prev_lvl_img = pyramid[-1]

	for img in range(len(pyramid)-1)[::-1]:

		prev_lvl_img_expand = expand_laplacian(prev_lvl_img)

		if pyramid[img].shape != prev_lvl_img_expand.shape:
			prev_lvl_img = pyramid[img] +prev_lvl_img_expand[:pyramid[img].shape[0],:pyramid[img].shape[1]]

		else:
			prev_lvl_img = pyramid[img] + prev_lvl_img_expand

	return prev_lvl_img

def burt_adelson(imgA, imgB, mask):

	gaussian_mask = compute_gaussian_pyramid(mask, levels=6)

	lA = compute_laplacian_pyramid(imgA)
	show(*lA)
	lB = compute_laplacian_pyramid(imgB)
	show(*lB)

	pyramid = combine_laplacian_pyramids(lA, lB, gaussian_mask)
	show(*pyramid)
	img_blend = collapse_laplacian_pyramid(pyramid)

	show(img_blend)
	return [img_blend]
	# return (maskA, maskB, gaussian_mask, lA, lB, pyramid, [img_blend])


def visualize_pyr(pyramid):

	"""Create a single image by vertically stacking the levels of a pyramid."""
	shape = np.atleast_3d(pyramid[0]).shape[:-1]  # need num rows & cols only
	img_stack = [cv2.resize(layer, shape[::-1],interpolation=3) for layer in pyramid]
	return np.vstack(img_stack).astype(np.uint8)




def blend_and_store_images(black_image, white_image, mask):

	"""Apply pyramid blending to each color channel of the input images """

	# Convert to double and normalize the images to the range [0..1]
	# to avoid arithmetic overflow issues
	b_img = np.atleast_3d(black_image).astype(np.float) / 255.
	w_img = np.atleast_3d(white_image).astype(np.float) / 255.
	m_img = np.atleast_3d(mask).astype(np.float) / 255.
	num_channels = black_image.shape[-1]

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

	show(visualize_pyr(stack))
		#cv2.imwrite('../images/outimg.png', visualize_pyr(stack))


A = cv2.imread("../images/apple.jpg",1)
A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
B = cv2.imread("../images/orange.jpg",1)
B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
mask = cv2.imread("../images/mask.jpg",1)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

#show(*compute_laplacian_pyramid(A))

blend_and_store_images(B,A,mask)
