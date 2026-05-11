import math
from typing import List

from uwimg import Image, make_image
import numpy as np
from src.hw1.process_image import get_pixel, set_pixel, hsv_to_rgb

TWOPI = 6.2831853


# ----------------------------- Resizing -----------------------------

def nn_interpolate(im: Image, x: float, y: float, c: int) -> float:
    # Performs nearest-neighbor interpolation at floating (x, y) for channel c.
    
    return get_pixel(im, round(x), round(y), c)


def nn_resize(im: Image, w: int, h: int) -> Image:
    # Uses nearest-neighbor interpolation to resize to (w, h).
    # Use float32 to match C reference implementation precision (avoids round() boundary mismatches).
    new_im = make_image(w, h, im.c)
    scale_x = np.float32(im.w) / np.float32(w)
    bias_x = (scale_x - np.float32(1)) / np.float32(2)
    scale_y = np.float32(im.h) / np.float32(h)
    bias_y = (scale_y - np.float32(1)) / np.float32(2)
    for c in range(im.c):
        for y in range(h):
            for x in range(w):
                fx = np.float32(x) * scale_x + bias_x
                fy = np.float32(y) * scale_y + bias_y
                new_im.data[c, y, x] = nn_interpolate(im, fx, fy, c)
    return new_im


def bilinear_interpolate(im: Image, x: float, y: float, c: int) -> float:
    # Performs bilinear interpolation at floating (x, y) for channel c.
    x1 = math.floor(x)
    y1 = math.floor(y)
    x2 = x1 + 1
    y2 = y1 + 1
    d1 = x-x1
    d2 = 1-d1
    d3 = y-y1
    d4 = 1-d3
    A1 = d2*d4
    A2 = d1*d4
    A3 = d2*d3
    A4 = d1*d3
    V1 = get_pixel(im, x1, y1, c)
    V2 = get_pixel(im, x2, y1, c)
    V3 = get_pixel(im, x1, y2, c)
    V4 = get_pixel(im, x2, y2, c)
    return A1*V1 + A2*V2 + A3*V3 + A4*V4


def bilinear_resize(im: Image, w: int, h: int) -> Image:
    # Use float32 to match C reference implementation precision.
    new_im = make_image(w, h, im.c)
    scale_x = np.float32(im.w) / np.float32(w)
    bias_x = (scale_x - np.float32(1)) / np.float32(2)
    scale_y = np.float32(im.h) / np.float32(h)
    bias_y = (scale_y - np.float32(1)) / np.float32(2)
    for c in range(im.c):
        for y in range(h):
            for x in range(w):
                fx = np.float32(x) * scale_x + bias_x
                fy = np.float32(y) * scale_y + bias_y
                new_im.data[c, y, x] = bilinear_interpolate(im, fx, fy, c)
    return new_im


# ----------------------------- Filtering -----------------------------

def l1_normalize(im: Image) -> None:
    # TODO
    # Divide each value by the sum of all values (in-place).
    sum_val = im.data.sum()
    if sum_val == 0:
        return
    im.data = im.data / sum_val
    return


def make_box_filter(w: int) -> Image:
    filter = make_image(w,w,1)
    filter.data = np.ones((1,w,w))
    l1_normalize(filter)
    # Make a (w x w x 1) filter filled with 1s, then l1_normalize.
    return filter


def convolve_image(im: Image, filt: Image, preserve: int) -> Image:
    # TODO
    # Convolve im with filt. preserve=1 keeps channels, else outputs 1 channel.
    # Must assert (im.c == filt.c) or (filt.c == 1).
    assert (im.c == filt.c) or (filt.c == 1)
    new_c = im.c if preserve == 1 else 1
    new_im = make_image(im.w, im.h, new_c)
    for c in range(im.c):
        filter_c = c if filt.c == im.c else 0
        out_c_idx = c if preserve == 1 else 0
        for y in range(im.h):
            for x in range(im.w):
                val = 0
                for fy in range(filt.h):
                    for fx in range(filt.w):
                        val += get_pixel(im, x + fx - (filt.w//2), y + fy - (filt.h//2), c) * get_pixel(filt, fx, fy, filter_c)
                new_im.data[out_c_idx, y, x] += val
    return new_im
        



def make_highpass_filter() -> Image:
    filter = make_image(3,3,1)
    filter.data = np.array([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]])
    return filter


def make_sharpen_filter() -> Image:
    filter = make_image(3,3,1)
    filter.data = np.array([[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]])
    return filter


def make_emboss_filter() -> Image:
    filter = make_image(3,3,1)
    filter.data = np.array([[[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]])
    return filter


# Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
# Answer: TODO

# Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
# Answer: TODO


def make_gaussian_filter(sigma: float) -> Image:
    # TODO
    # Kernel size is next highest odd integer from 6*sigma (matches C: ((int)(6*sigma))|1)
    # Fill using 2D gaussian, then l1_normalize.
    ksize = int(6*sigma) | 1
    filter = make_image(ksize, ksize, 1)
    for y in range(ksize):
        for x in range(ksize):
            filter.data[0, y, x] = math.exp(-((x - ksize//2)**2 + (y - ksize//2)**2) / (2 * sigma**2))
    l1_normalize(filter)
    return filter


def add_image(a: Image, b: Image) -> Image:
    assert a.data.shape == b.data.shape
    new_im = make_image(a.w, a.h, a.c)
    new_im.data = a.data + b.data
    return new_im


def sub_image(a: Image, b: Image) -> Image:
    assert a.data.shape == b.data.shape
    new_im = make_image(a.w, a.h, a.c)
    new_im.data = a.data - b.data
    return new_im


def make_gx_filter() -> Image:
    # TODO
    # Create a 3x3 Sobel Gx filter.
    filter = make_image(3,3,1)
    filter.data = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
    return filter


def make_gy_filter() -> Image:
    # TODO
    # Create a 3x3 Sobel Gy filter.
    filter = make_image(3,3,1)
    filter.data = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
    return filter


def feature_normalize(im: Image) -> None:
    # TODO
    # Normalize to [0,1] using (x-min)/(max-min); if max==min set all to 0.
    min_val = np.min(im.data)
    max_val = np.max(im.data)
    if max_val == min_val:
        im.data = np.zeros_like(im.data)
        return
    im.data = (im.data - min_val) / (max_val - min_val)
    return


def sobel_image(im: Image) -> List[Image]:
    # TODO
    # Return [magnitude, direction] as two 1-channel images.
    gxf = make_gx_filter()
    gyf = make_gy_filter()
    gx = convolve_image(im, gxf, 0)
    gy = convolve_image(im, gyf, 0)
    magnitude = make_image(im.w, im.h, 1)
    direction = make_image(im.w, im.h, 1)
    magnitude.data = np.sqrt(gx.data**2 + gy.data**2)
    direction.data = np.arctan2(gy.data, gx.data)
    return [magnitude, direction]


def colorize_sobel(im: Image) -> Image:
    # TODO
    # Use sobel magnitude as S and V, direction as H, then hsv_to_rgb.
    magnitude, direction = sobel_image(im)
    hsv = make_image(im.w, im.h, 3)
    hsv.data[0] = direction
    hsv.data[1] = magnitude
    hsv.data[2] = magnitude
    hsv_to_rgb(hsv)
    return hsv


# EXTRA CREDIT: Median filter

"""
def apply_median_filter(im: Image, kernel_size: int) -> Image:
    return make_image(1, 1, 1)
"""

# SUPER EXTRA CREDIT: Bilateral filter

"""
def apply_bilateral_filter(im: Image, sigma1: float, sigma2: float) -> Image:
    return make_image(1, 1, 1)
"""