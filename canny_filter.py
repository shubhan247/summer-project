import cv2
import numpy as np
from scipy import ndimage

def norm(img1,img2):
	img_copy = np.zeros(img1.shape)
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			q = (img1[i][j]**2 + img2[i][j]**2)**(1/2)
			if(q>90):
				img_copy[i][j] = 255
			else:
				img_copy[i][j] = 0
	return img_copy
def sobel_filters(image):
	img_new = image
	Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
	Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
	print(Kx)
	Ix = conv(image,Kx)
	Iy = conv(image,Ky)
	#G = np.hypot(Ix, Iy)
	#G = G / G.max() * 255
	#theta = np.arctan2(Iy, Ix)
	g_sobel = norm(Ix,Iy)
	return g_sobel
def conv_transform(image):
	image_copy = image
	
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
	return image_copy
def gaussian_kernel(size,sigma=1):
	size = int(size)//2
	x,y = np.mgrid[-size:size+1,-size:size+1]
	normal = 1 / (2.0 * np.pi * sigma**2)
	g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
	return g
def conv(image,kernel):
	kernel = conv_transform(kernel)
	image_h = image.shape[0]
	image_w = image.shape[1]
	kernel_h = kernel.shape[0]
	kernel_w = kernel.shape[1]
	h = kernel_h//2
	w = kernel_w//2
	image_conv = image
	for i in range(h,image_h-h):
		for j in range(w,image_w-w):
			sum = 0
			for m in range(kernel_h):
				for n in range(kernel_w):
					sum = sum + kernel[m][n]*image[i-h+m][j-w+n]
			image_conv[i][j] = sum
	return image_conv


img = cv2.imread('hp.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel = np.array([])
kernel = gaussian_kernel(5)
blur = conv(gray,kernel)
blur2 = sobel_filters(blur)
#cv2.imwrite('blur.jpg',blur)
#blur2 = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
cv2.imshow('image',blur2)


cv2.waitKey(0)
cv2.destroyAllWindows()    
