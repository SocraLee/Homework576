import math
from uwimg import Image, make_image,save_image,load_image
import numpy as np

def get_pixel(im: Image, x: int, y: int, c: int) -> float:
    # padding the exceeding index
    ic,ih,iw = im.data.shape
    if x < 0 :
        x = 0
    if y < 0 :
        y = 0
    if c < 0 :
        c = 0
    if x >= iw :
        x = iw - 1
    if y >= ih :
        y = ih - 1
    if c >= ic :
        c = ic - 1
    return im.data[c,y,x]


def set_pixel(im: Image, x: int, y: int, c: int, v: float) -> None:
    ic,ih,iw = im.data.shape
    if x < 0 :
        x = 0
    if y < 0 :
        y = 0
    if c < 0 :
        c = 0
    if x >= iw :
        x = iw - 1
    if y >= ih :
        y = ih - 1
    if c >= ic :
        c = ic - 1
    im.data[c,y,x] = v
    return


def copy_image(im: Image) -> Image:
    copy = make_image(im.w, im.h, im.c)
    copy.data = im.data.copy()
    return copy


def rgb_to_grayscale(im: Image) -> Image:
    assert im.c == 3
    gray = make_image(im.w, im.h, 1)
    gray.data[0] = im.data[0]*0.299 + im.data[1]*0.587 + im.data[2]*0.114
    return gray


def shift_image(im: Image, c: int, v: float) -> None:
    im.data[c] = im.data[c] + v
    return


def clamp_image(im: Image) -> None:
    im.data = np.clip(im.data, 0, 1)
    return


# These might be handy
def three_way_max(a: float, b: float, c: float) -> float:
    return (a if a > b else b) if ((a if a > b else b) > c) else c


def three_way_min(a: float, b: float, c: float) -> float:
    return (a if a < b else b) if ((a if a < b else b) < c) else c


def rgb_to_hsv(im: Image) -> None:
    R = im.data[0].copy()
    G = im.data[1].copy()
    B = im.data[2].copy()

    V = np.maximum(R, np.maximum(G, B))
    m = np.minimum(R, np.minimum(G, B))
    C = V - m
    S = np.where(V == 0, 0, C / V)

    H = np.zeros_like(R)# default by 0
    H = np.where((V == R) & (C != 0), (G - B) / C, H)
    H = np.where((V == G) & (C != 0), (B - R) / C + 2, H)
    H = np.where((V == B) & (C != 0), (R - G) / C + 4, H)
    H = H / 6
    H = np.where(H < 0, H + 1, H)
    H = np.clip(H, 0, 1)# wrap H by [0,1)]

    im.data[0] = H
    im.data[1] = S
    im.data[2] = V
    return


def hsv_to_rgb(im: Image) -> None:

    H = im.data[0].copy()
    S = im.data[1].copy()
    V = im.data[2].copy()
    
    H = H * 6
    Hi = np.floor(H).astype(int)
    F = H - Hi
    P = V * (1 - S)
    Q = V * (1 - F * S)
    T = V * (1 - (1 - F) * S)
    
    im.data[0] = np.where(Hi == 0, V, im.data[0])
    im.data[0] = np.where(Hi == 1, Q, im.data[0])
    im.data[0] = np.where(Hi == 2, P, im.data[0])
    im.data[0] = np.where(Hi == 3, P, im.data[0])
    im.data[0] = np.where(Hi == 4, T, im.data[0])
    im.data[0] = np.where(Hi == 5, V, im.data[0])
    
    im.data[1] = np.where(Hi == 0, T, im.data[1])
    im.data[1] = np.where(Hi == 1, V, im.data[1])
    im.data[1] = np.where(Hi == 2, V, im.data[1])
    im.data[1] = np.where(Hi == 3, Q, im.data[1])
    im.data[1] = np.where(Hi == 4, P, im.data[1])
    im.data[1] = np.where(Hi == 5, P, im.data[1])
    
    im.data[2] = np.where(Hi == 0, P, im.data[2])
    im.data[2] = np.where(Hi == 1, P, im.data[2])
    im.data[2] = np.where(Hi == 2, T, im.data[2])
    im.data[2] = np.where(Hi == 3, V, im.data[2])
    im.data[2] = np.where(Hi == 4, V, im.data[2])
    im.data[2] = np.where(Hi == 5, Q, im.data[2])
    
    return

def scale_image(im: Image, c: int, v: float) -> None:
    im.data[c] = im.data[c] * v
    return