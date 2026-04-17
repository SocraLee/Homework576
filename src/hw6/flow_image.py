import math

from src.hw1.process_image import get_pixel, set_pixel, rgb_to_grayscale, copy_image
from src.hw2.modify_image import (
	nn_resize,
	convolve_image,
	sub_image,
	make_gx_filter,
	make_gy_filter,
)
from src.hw3.harris_image import smooth_image
from uwimg import make_image, free_image
import src.matrix as matrix

TWOPI = 6.2831853


def open_video_stream(*args):
	raise NotImplementedError("Must compile with OpenCV")


def get_image_from_stream(*args):
	raise NotImplementedError("Must compile with OpenCV")


def show_image(*args):
	raise NotImplementedError("Must compile with OpenCV")


# Draws a line on an image with color corresponding to the direction of line
# image im: image to draw line on
# float x, y: starting point of line
# float dx, dy: vector corresponding to line angle and magnitude
def draw_line(im, x, y, dx, dy):
	assert im.c == 3
	angle = 6 * (math.atan2(dy, dx) / TWOPI + 0.5)
	index = int(math.floor(angle))
	f = angle - index
	r = g = b = 0.0
	if index == 0:
		r = 1
		g = f
		b = 0
	elif index == 1:
		r = 1 - f
		g = 1
		b = 0
	elif index == 2:
		r = 0
		g = 1
		b = f
	elif index == 3:
		r = 0
		g = 1 - f
		b = 1
	elif index == 4:
		r = f
		g = 0
		b = 1
	else:
		r = 1
		g = 0
		b = 1 - f

	d = math.sqrt(dx * dx + dy * dy)
	i = 0.0
	while i < d:
		xi = int(x + dx * i / d)
		yi = int(y + dy * i / d)
		set_pixel(im, xi, yi, 0, r)
		set_pixel(im, xi, yi, 1, g)
		set_pixel(im, xi, yi, 2, b)
		i += 1


# Make an integral image or summed area table from an image
# image im: image to process
# returns: image I such that I[x,y] = sum{i<=x, j<=y}(im[i,j])
def make_integral_image(im):
	integ = make_image(im.w, im.h, im.c)
	# TODO: fill in the integral image
	return integ


# Apply a box filter to an image using an integral image for speed
# image im: image to smooth
# int s: window size for box filter
# returns: smoothed image
def box_filter_image(im, s):
	integ = make_integral_image(im)
	S = make_image(im.w, im.h, im.c)
	# TODO: fill in S using the integral image.
	return S


# Calculate the time-structure matrix of an image pair.
# image im: the input image.
# image prev: the previous image in sequence.
# int s: window size for smoothing.
# returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
#          3rd channel is IxIy, 4th channel is IxIt, 5th channel is IyIt.
def time_structure_matrix(im, prev, s):
	converted = 0
	if im.c == 3:
		converted = 1
		im = rgb_to_grayscale(im)
		prev = rgb_to_grayscale(prev)

	# TODO: calculate gradients, structure components, and smooth them

	S = None

	if converted:
		free_image(im)
		free_image(prev)
	return S


# Calculate the velocity given a structure image
# image S: time-structure image
# int stride: only calculate subset of pixels for speed
def velocity_image(S, stride):
	v = make_image(S.w // stride, S.h // stride, 3)
	for j in range((stride - 1) // 2, S.h, stride):
		for i in range((stride - 1) // 2, S.w, stride):
			Ixx = float(S.data.reshape(-1)[i + S.w * j + 0 * S.w * S.h])
			Iyy = float(S.data.reshape(-1)[i + S.w * j + 1 * S.w * S.h])
			Ixy = float(S.data.reshape(-1)[i + S.w * j + 2 * S.w * S.h])
			Ixt = float(S.data.reshape(-1)[i + S.w * j + 3 * S.w * S.h])
			Iyt = float(S.data.reshape(-1)[i + S.w * j + 4 * S.w * S.h])

			# TODO: calculate vx and vy using the flow equation
			vx = 0
			vy = 0

			set_pixel(v, i // stride, j // stride, 0, vx)
			set_pixel(v, i // stride, j // stride, 1, vy)
	return v


# Draw lines on an image given the velocity
# image im: image to draw on
# image v: velocity of each pixel
# float scale: scalar to multiply velocity by for drawing
def draw_flow(im, v, scale):
	stride = im.w // v.w
	for j in range((stride - 1) // 2, im.h, stride):
		for i in range((stride - 1) // 2, im.w, stride):
			dx = scale * get_pixel(v, i // stride, j // stride, 0)
			dy = scale * get_pixel(v, i // stride, j // stride, 1)
			if abs(dx) > im.w:
				dx = 0
			if abs(dy) > im.h:
				dy = 0
			draw_line(im, i, j, dx, dy)


# Constrain the absolute value of each image pixel
# image im: image to constrain
# float v: each pixel will be in range [-v, v]
def constrain_image(im, v):
	flat = im.data.reshape(-1)
	for i in range(im.w * im.h * im.c):
		if flat[i] < -v:
			flat[i] = -v
		if flat[i] > v:
			flat[i] = v


# Calculate the optical flow between two images
# image im: current image
# image prev: previous image
# int smooth: amount to smooth structure matrix by
# int stride: downsampling for velocity matrix
# returns: velocity matrix
def optical_flow_images(im, prev, smooth, stride):
	S = time_structure_matrix(im, prev, smooth)
	v = velocity_image(S, stride)
	constrain_image(v, 6)
	vs = smooth_image(v, 2)
	free_image(v)
	free_image(S)
	return vs


# Run optical flow demo on webcam
# int smooth: amount to smooth structure matrix by
# int stride: downsampling for velocity matrix
# int div: downsampling factor for images from webcam
def optical_flow_webcam(smooth, stride, div):
	cap = open_video_stream(0, 0, 1280, 720, 30)
	prev = get_image_from_stream(cap)
	prev_c = nn_resize(prev, prev.w // div, prev.h // div)
	im = get_image_from_stream(cap)
	im_c = nn_resize(im, im.w // div, im.h // div)
	while im.data is not None:
		copy = copy_image(im)
		v = optical_flow_images(im_c, prev_c, smooth, stride)
		draw_flow(copy, v, smooth * div)
		key = show_image(copy, "flow", 5)
		free_image(v)
		free_image(copy)
		free_image(prev)
		free_image(prev_c)
		prev = im
		prev_c = im_c
		if key != -1:
			key = key % 256
			print(f"{key}")
			if key == 27:
				break
		im = get_image_from_stream(cap)
		im_c = nn_resize(im, im.w // div, im.h // div)
