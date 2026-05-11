import math
import numpy as np
from typing import List
from src.hw1.process_image import get_pixel, set_pixel, copy_image
from src.hw2.modify_image import make_gaussian_filter, make_gx_filter, make_gy_filter, convolve_image, l1_normalize
from uwimg import make_image

class Point:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

class Descriptor:
    def __init__(self):
        self.p = Point()
        self.data = None
        self.n = 0

# Create a feature descriptor for an index in an image.
# image im: source image.
# int i: index in image for the pixel we want to describe.
# returns: descriptor for that index.
def describe_index(im, i: int) -> Descriptor:
    w = 5
    d = Descriptor()
    d.p.x = i % im.w
    d.p.y = i // im.w
    d.data = np.zeros(w * w * im.c, dtype=np.float32)
    d.n = w * w * im.c
    count = 0
    # If you want you can experiment with other descriptors
    # This subtracts the central value from neighbors
    # to compensate some for exposure/lighting changes.
    for c in range(im.c):
        cval = im.data[c, i // im.w, i % im.w]
        for dx in range(-(w // 2), (w + 1) // 2):
            for dy in range(-(w // 2), (w + 1) // 2):
                val = get_pixel(im, (i % im.w) + dx, (i // im.w) + dy, c)
                d.data[count] = cval - val
                count += 1
    return d

# Marks the spot of a point in an image.
# image im: image to mark.
# ponit p: spot to mark in the image.
def mark_spot(im, p: Point) -> None:
    x = int(p.x)
    y = int(p.y)
    for i in range(-9, 10):
        set_pixel(im, x + i, y, 0, 1.0)
        set_pixel(im, x, y + i, 0, 1.0)
        set_pixel(im, x + i, y, 1, 0.0)
        set_pixel(im, x, y + i, 1, 0.0)
        set_pixel(im, x + i, y, 2, 1.0)
        set_pixel(im, x, y + i, 2, 1.0)

# Marks corners denoted by an array of descriptors.
# image im: image to mark.
# descriptor *d: corners in the image.
# int n: number of descriptors to mark.
def mark_corners(im, d: List[Descriptor]) -> None:
    for i in range(len(d)):
        mark_spot(im, d[i].p)

# Creates a 1d Gaussian filter.
# float sigma: standard deviation of Gaussian.
# returns: single row image of the filter.
def make_1d_gaussian(sigma: float):
    ksize = int(sigma * 6) | 1
    g = make_image(ksize, 1, 1)
    for i in range(ksize):
        g.data[0, 0, i] = math.exp(-(i - ksize//2)**2 / (2 * sigma**2))
    l1_normalize(g)
    return g


# Smooths an image using separable Gaussian filter.
# image im: image to smooth.
# float sigma: std dev. for Gaussian.
# returns: smoothed image.
def smooth_image(im, sigma: float):
    g = make_1d_gaussian(sigma)
    tmp = convolve_image(im, g, 1)
    g_T = make_image(1,g.w,1)
    g_T.data = g.data.reshape(1,g.w,1)
    im = convolve_image(tmp, g_T, 1)
    return im

# Calculate the structure matrix of an image.
# image im: the input image.
# float sigma: std dev. to use for weighted sum.
# returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
#          third channel is IxIy.
def structure_matrix(im, sigma):
    gx = make_gx_filter()
    gy = make_gy_filter()
    ix = convolve_image(im, gx, 0)     
    iy = convolve_image(im, gy, 0)
    S = make_image(im.w, im.h, 3)
    S.data[0] = ix.data[0] * ix.data[0]  
    S.data[1] = iy.data[0] * iy.data[0]
    S.data[2] = ix.data[0] * iy.data[0]    
    S = smooth_image(S, sigma)              
    return S


# Estimate the cornerness of each pixel given a structure matrix S.
# image S: structure matrix for an image.
# returns: a response map of cornerness calculations.
def cornerness_response(S):
    R = make_image(S.w, S.h, 1)
    # TODO: fill in R, "cornerness" for each pixel using the structure matrix.
    # We'll use formulation det(S) - alpha * trace(S)^2, alpha = .06.
    det = S.data[0] * S.data[1] - S.data[2] * S.data[2]
    trace = S.data[0] + S.data[1]
    R.data[0] = det - 0.06 * trace * trace
    return R

# Perform non-max supression on an image of feature responses.
# image im: 1-channel image of feature responses.
# int w: distance to look for larger responses.
# returns: image with only local-maxima responses within w pixels.
def nms_image(im, w: int):
    r = copy_image(im)
    # TODO: perform NMS on the response map.
    # for every pixel in the image:
    #     for neighbors within w:
    #         if neighbor response greater than pixel response:
    #             set response to be very low (I use -999999 [why not 0??])
    for y in range(im.h):
        for x in range(im.w):
            for i in range(-w, w + 1):
                for j in range(-w, w + 1):
                    nx = x + i
                    ny = y + j
                    if nx < 0 or nx >= im.w or ny < 0 or ny >= im.h:
                        continue
                    if im.data[0, ny, nx] > r.data[0, y, x]:
                        r.data[0, y, x] = -999999
                        break
                if r.data[0, y, x] == -999999:
                    break
    return r

# Perform harris corner detection and extract features from the corners.
# image im: input image.
# float sigma: std. dev for harris.
# float thresh: threshold for cornerness.
# int nms: distance to look for local-maxes in response map.
# returns: array of descriptors of the corners in the image.
def harris_corner_detector(im, sigma: float, thresh: float, nms: int) -> List[Descriptor]:
    # Calculate structure matrix
    S = structure_matrix(im, sigma)

    # Estimate cornerness
    R = cornerness_response(S)

    # Run NMS on the responses
    Rnms = nms_image(R, nms)

    # TODO: count number of responses over threshold
    mask = Rnms.data[0] > thresh
    count = np.sum(mask)
    ys, xs = np.nonzero(mask)
    
    d = [Descriptor() for _ in range(count)]
    idx = 0
    for y, x in zip(ys, xs):
        d[idx] = describe_index(im, y * im.w + x)
        idx += 1
    return d

# Find and draw corners on an image.
# image im: input image.
# float sigma: std. dev for harris.
# float thresh: threshold for cornerness.
# int nms: distance to look for local-maxes in response map.
def detect_and_draw_corners(im, sigma: float, thresh: float, nms: int) -> None:
    d = harris_corner_detector(im, sigma, thresh, nms)
    mark_corners(im, d)