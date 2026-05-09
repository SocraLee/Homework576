import math
import random
import numpy as np
from typing import List
from src.hw1.process_image import get_pixel, set_pixel, copy_image
from src.hw2.modify_image import bilinear_interpolate
from uwimg import Image, make_image, save_image
from src.hw3.harris_image import Point, Descriptor, harris_corner_detector, mark_corners

class Match:
    def __init__(self):
        self.p = Point()
        self.q = Point()
        self.ai = 0
        self.bi = 0
        self.distance = 0.0

# Comparator for matches
# Match a, b: objects to compare.
# returns: result of comparison, 0 if same, 1 if a > b, -1 if a < b.
def match_compare(a: Match, b: Match) -> int:
    if a.distance < b.distance: return -1
    elif a.distance > b.distance: return 1
    else: return 0

# Helper function to create 2d points.
# float x, y: coordinates of point.
# returns: the point.
def make_point(x: float, y: float) -> Point:
    p = Point()
    p.x = x
    p.y = y
    return p

# Place two images side by side on canvas, for drawing matching pixels.
# image a, b: images to place.
# returns: image with both a and b side-by-side.
def both_images(a: Image, b: Image):
    both = make_image(a.w + b.w, max(a.h, b.h), max(a.c, b.c))
    for k in range(a.c):
        for j in range(a.h):
            for i in range(a.w):
                set_pixel(both, i, j, k, get_pixel(a, i, j, k))

    for k in range(b.c):
        for j in range(b.h):
            for i in range(b.w):
                set_pixel(both, i + a.w, j, k, get_pixel(b, i, j, k))

    return both

# Draws lines between matching pixels in two images.
# image a, b: two images that have matches.
# List[Match] matches: list of matches between a and b.
# int n: number of matches.
# int inliers: number of inliers at beginning of matches, drawn in green.
# returns: image with matches drawn between a and b on same canvas.
def draw_matches(a: Image, b: Image, matches: List[Match], n: int, inliers: int):
    both = both_images(a, b)
    for i in range(n):
        bx = int(matches[i].p.x)
        ex = int(matches[i].q.x)
        by = int(matches[i].p.y)
        ey = int(matches[i].q.y)
        for j in range(bx, ex + a.w):
            r = int(float(j - bx) / (ex + a.w - bx) * (ey - by) + by)
            set_pixel(both, j, r, 0, 0.0 if i < inliers else 1.0)
            set_pixel(both, j, r, 1, 1.0 if i < inliers else 0.0)
            set_pixel(both, j, r, 2, 0.0)
    return both

# Draw the matches with inliers in green between two images.
# image a, b: two images to match.
# List[Match] m: matches list
def draw_inliers(a: Image, b: Image, H: np.ndarray, m: List[Match], n: int, thresh: float):
    inliers = model_inliers(H, m, n, thresh)
    lines = draw_matches(a, b, m, n, inliers)
    return lines

# Find corners, match them, and draw them between two images.
# image a, b: images to match.
# float sigma: gaussian for harris corner detector. Typical: 2
# float thresh: threshold for corner/no corner. Typical: 1-5
# int nms: window to perform nms on. Typical: 3
def find_and_draw_matches(a: Image, b: Image, sigma: float, thresh: float, nms: int):
    mn = [0]
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)
    m = match_descriptors(ad, len(ad), bd, len(bd), mn)

    mark_corners(a, ad)
    mark_corners(b, bd)
    lines = draw_matches(a, b, m, mn[0], 0)
    return lines

# Calculates L1 distance between to floating point arrays.
# float array a, b: arrays to compare.
# int n: number of values in each array.
# returns: l1 distance between arrays (sum of absolute differences).
def l1_distance(a: np.ndarray, b: np.ndarray, n: int) -> float:
    # TODO: return the correct number.
    return 0.0

# Finds best matches between descriptors of two images.
# List[Descriptor] a, b: list of descriptors for pixels in two images.
# int an, bn: number of descriptors in arrays a and b.
# List[int] mn: single-element list to hold number of matches found.
# returns: best matches found. each descriptor in a should match with at most
#          one other descriptor in b.
def match_descriptors(a: List[Descriptor], an: int, b: List[Descriptor], bn: int, mn: List[int]) -> List[Match]:
    # We will have at most an matches.
    mn[0] = an
    m = [Match() for _ in range(an)]
    for j in range(an):
        # TODO: for every descriptor in a, find best match in b.
        # record ai as the index in a and bi as the index in b.
        bind = 0 # <- find the best match
        m[j].ai = j
        m[j].bi = bind # <- should be index in b.
        m[j].p = a[j].p
        m[j].q = b[bind].p
        m[j].distance = 0.0 # <- should be the smallest L1 distance!

    count = 0
    seen = [0] * bn
    # TODO: we want matches to be injective (one-to-one).
    # Sort matches based on distance using match_compare.
    # Then throw out matches to the same element in b. Use seen to keep track.
    # Each point should only be a part of one match.
    # Some points will not be in a match.
    # In practice just bring good matches to front of list, set mn[0].
    mn[0] = count
    return m

# Apply a projective transformation to a point.
# np.ndarray H: homography to project point.
# point p: point to project.
# returns: point projected using the homography.
def project_point(H: np.ndarray, p: Point) -> Point:
    # TODO: project point p with homography H.
    # Remember that homogeneous coordinates are equivalent up to scalar.
    # Have to divide by.... something...
    q = make_point(0, 0)
    return q

# Calculate L2 distance between two points.
# point p, q: points.
# returns: L2 distance between them.
def point_distance(p: Point, q: Point) -> float:
    # TODO: should be a quick one.
    return 0.0

# Count number of inliers in a set of matches. Should also bring inliers
# to the front of the array.
# np.ndarray H: homography between coordinate systems.
# List[Match] m: matches to compute inlier/outlier.
# int n: number of matches in m.
# float thresh: threshold to be an inlier.
# returns: number of inliers whose projected point falls within thresh of
#          their match in the other image. Should also rearrange matches
#          so that the inliers are first in the array. For drawing.
def model_inliers(H: np.ndarray, m: List[Match], n: int, thresh: float) -> int:
    count = 0
    # TODO: count number of matches that are inliers
    # i.e. distance(H*p, q) < thresh
    # Also, sort the matches m so the inliers are the first 'count' elements.
    return count

# Randomly shuffle matches for RANSAC.
# List[Match] m: matches to shuffle in place.
# int n: number of elements in matches.
def randomize_matches(m: List[Match], n: int) -> None:
    # TODO: implement Fisher-Yates to shuffle the array.
    pass

# Computes homography between two images given matching pixels.
# List[Match] matches: matching points between images.
# int n: number of matches to use in calculating homography.
# returns: matrix representing homography H that maps image a to image b.
def compute_homography(matches: List[Match], n: int) -> np.ndarray:
    M = np.zeros((n * 2, 8), dtype=np.float64)
    b = np.zeros((n * 2, 1), dtype=np.float64)

    for i in range(n):
        x  = matches[i].p.x
        xp = matches[i].q.x
        y  = matches[i].p.y
        yp = matches[i].q.y
        # TODO: fill in the matrices M and b.

    # TODO: solve system M a = b
    # If a solution can't be found, return None

    H = np.zeros((3, 3), dtype=np.float64)
    # TODO: fill in the homography H based on the result in a.

    return H

# Perform RANdom SAmple Consensus to calculate homography for noisy matches.
# List[Match] m: set of matches.
# int n: number of matches.
# float thresh: inlier/outlier distance threshold.
# int k: number of iterations to run.
# int cutoff: inlier cutoff to exit early.
# returns: matrix representing most common homography between matches.
def RANSAC(m: List[Match], n: int, thresh: float, k: int, cutoff: int) -> np.ndarray:
    best = 0
    Hb = np.eye(3, dtype=np.float64)
    Hb[0, 2] = 256.0
    Hb[1, 2] = 0.0
    # TODO: fill in RANSAC algorithm.
    # for k iterations:
    #     shuffle the matches
    #     compute a homography with a few matches (how many??)
    #     if new homography is better than old (how can you tell?):
    #         compute updated homography using all inliers
    #         remember it and how good it is
    #         if it's better than the cutoff:
    #             return it immediately
    # if we get to the end return the best homography
    return Hb

# Stitches two images together using a projective transformation.
# image a, b: images to stitch.
# np.ndarray H: homography from image a coordinates to image b coordinates.
# returns: combined image stitched together.
def combine_images(a: Image, b: Image, H: np.ndarray):
    Hinv = np.linalg.inv(H)

    # Project the corners of image b into image a coordinates.
    c1 = project_point(Hinv, make_point(0, 0))
    c2 = project_point(Hinv, make_point(b.w - 1, 0))
    c3 = project_point(Hinv, make_point(0, b.h - 1))
    c4 = project_point(Hinv, make_point(b.w - 1, b.h - 1))

    # Find top left and bottom right corners of image b warped into image a.
    botright = make_point(max(c1.x, c2.x, c3.x, c4.x), max(c1.y, c2.y, c3.y, c4.y))
    topleft = make_point(min(c1.x, c2.x, c3.x, c4.x), min(c1.y, c2.y, c3.y, c4.y))

    # Find how big our new image should be and the offsets from image a.
    dx = int(min(0, topleft.x))
    dy = int(min(0, topleft.y))
    w = int(max(a.w, botright.x) - dx)
    h = int(max(a.h, botright.y) - dy)

    # Can disable this if you are making very big panoramas.
    # Usually this means there was an error in calculating H.
    if w > 7000 or h > 7000:
        print("output too big, stopping")
        return copy_image(a)

    c = make_image(w, h, a.c)

    # Paste image a into the new image offset by dx and dy.
    for k in range(a.c):
        for j in range(a.h):
            for i in range(a.w):
                pass
                # TODO: fill in.

    # TODO: Paste in image b as well.
    # You should loop over some points in the new image (which? all?)
    # and see if their projection from a coordinates to b coordinates falls
    # inside of the bounds of image b. If so, use bilinear interpolation to
    # estimate the value of b at that projection, then fill in image c.

    return c

# Create a panoramam between two images.
# image a, b: images to stitch together.
# float sigma: gaussian for harris corner detector. Typical: 2
# float thresh: threshold for corner/no corner. Typical: 1-5
# int nms: window to perform nms on. Typical: 3
# float inlier_thresh: threshold for RANSAC inliers. Typical: 2-5
# int iters: number of RANSAC iterations. Typical: 1,000-50,000
# int cutoff: RANSAC inlier cutoff. Typical: 10-100
# int draw: flag to draw inliers.
def panorama_image(a: Image, b: Image, sigma: float = 2, thresh: float = 5, nms: int = 3, inlier_thresh: float = 3, iters: int = 10000, cutoff: int = 100, draw: int = 0):
    random.seed(10)
    mn = [0]

    # Calculate corners and descriptors
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)

    # Find matches
    m = match_descriptors(ad, len(ad), bd, len(bd), mn)

    # Run RANSAC to find the homography
    H = RANSAC(m, mn[0], inlier_thresh, iters, cutoff)

    if draw:
        # Mark corners and matches between images
        mark_corners(a, ad)
        mark_corners(b, bd)
        inlier_matches = draw_inliers(a, b, H, m, mn[0], inlier_thresh)
        save_image(inlier_matches, "output/inliers")

    # Stitch the images together with the homography
    comb = combine_images(a, b, H)
    return comb

# Project an image onto a cylinder.
# image im: image to project.
# float f: focal length used to take image (in pixels).
# returns: image projected onto cylinder, then flattened.
def cylindrical_project(im: Image, f: float):
    # TODO: project image onto a cylinder
    c = copy_image(im)
    return c