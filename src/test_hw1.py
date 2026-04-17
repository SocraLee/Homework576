import math
import os

import uwimg
from uwimg import load_image, make_image, free_image
from src.hw1.process_image import (
    get_pixel,
    set_pixel,
    copy_image,
    rgb_to_grayscale,
    rgb_to_hsv,
    hsv_to_rgb,
    shift_image,
    clamp_image,
)

EPS = 0.002

def _gt(path: str) -> str:
    """Return .pillow variant when visionlib is unavailable."""
    if uwimg._vision is None:
        base, ext = os.path.splitext(path)
        alt = base + ".pillow" + ext
        if os.path.exists(alt):
            return alt
    return path

tests_total = 0
tests_fail = 0


def within_eps(a: float, b: float, eps: float) -> bool:
    # C: return a-eps<b && b<a+eps;
    return (a - eps) < b and b < (a + eps)


def same_image(a, b, eps: float) -> int:
    """
    EXACT port of C same_image(image a, image b, float eps)
    Return 1/0 like C.
    NOTE: eps is updated inside loop exactly like C (eps = thresh if thresh > eps).
    """
    if a.w != b.w or a.h != b.h or a.c != b.c:
        return 0

    # C memory is CHW flattened: index i corresponds to a.data[i]
    a_flat = a.data.reshape(-1)
    b_flat = b.data.reshape(-1)

    n = a.w * a.h * a.c
    for i in range(n):
        thresh = (abs(float(b_flat[i])) + abs(float(a_flat[i]))) * eps / 2.0
        if thresh > eps:
            eps = thresh
        if not within_eps(float(a_flat[i]), float(b_flat[i]), eps):
            # C prints:
            # printf("The value should be %f, but it is %f! \n", b.data[i], a.data[i]);
            print(f"The value should be {float(b_flat[i])}, but it is {float(a_flat[i])}! ")
            return 0
    return 1


def TEST(ex: bool, func: str, expr_str: str, file: str, line: int):
    """
    EXACT behavior of TEST macro in test.h:
      ++tests_total;
      if(!(EX)) {
        fprintf(stderr, "failed: [%s] testing [%s] in %s, line %d\n", __FUNCTION__, #EX, __FILE__, __LINE__);
        ++tests_fail;
      }
    """
    global tests_total, tests_fail
    tests_total += 1
    if not ex:
        tests_fail += 1
        print(f"failed: [{func}] testing [{expr_str}] in {file}, line {line}")


# ---------------- HW1 tests (exact order) ----------------

def test_get_pixel():
    im = load_image("data/dots.png")

    # Test within image
    TEST(within_eps(0.0, float(get_pixel(im, 0, 0, 0)), EPS),
         "test_get_pixel", "within_eps(0, get_pixel(im, 0,0,0), EPS)", "src_py/test_hw1.py", 0)
    TEST(within_eps(1.0, float(get_pixel(im, 1, 0, 1)), EPS),
         "test_get_pixel", "within_eps(1, get_pixel(im, 1,0,1), EPS)", "src_py/test_hw1.py", 0)
    TEST(within_eps(0.0, float(get_pixel(im, 2, 0, 1)), EPS),
         "test_get_pixel", "within_eps(0, get_pixel(im, 2,0,1), EPS)", "src_py/test_hw1.py", 0)

    # Test padding (clamp)
    TEST(within_eps(1.0, float(get_pixel(im, 0, 3, 1)), EPS),
         "test_get_pixel", "within_eps(1, get_pixel(im, 0,3,1), EPS)", "src_py/test_hw1.py", 0)
    TEST(within_eps(1.0, float(get_pixel(im, 7, 8, 0)), EPS),
         "test_get_pixel", "within_eps(1, get_pixel(im, 7,8,0), EPS)", "src_py/test_hw1.py", 0)
    TEST(within_eps(0.0, float(get_pixel(im, 7, 8, 1)), EPS),
         "test_get_pixel", "within_eps(0, get_pixel(im, 7,8,1), EPS)", "src_py/test_hw1.py", 0)
    TEST(within_eps(1.0, float(get_pixel(im, 7, 8, 2)), EPS),
         "test_get_pixel", "within_eps(1, get_pixel(im, 7,8,2), EPS)", "src_py/test_hw1.py", 0)

    free_image(im)


def test_set_pixel():
    gt = load_image("data/dots.png")
    d = make_image(4, 2, 3)

    set_pixel(d, 0, 0, 0, 0); set_pixel(d, 0, 0, 1, 0); set_pixel(d, 0, 0, 2, 0)
    set_pixel(d, 1, 0, 0, 1); set_pixel(d, 1, 0, 1, 1); set_pixel(d, 1, 0, 2, 1)
    set_pixel(d, 2, 0, 0, 1); set_pixel(d, 2, 0, 1, 0); set_pixel(d, 2, 0, 2, 0)
    set_pixel(d, 3, 0, 0, 1); set_pixel(d, 3, 0, 1, 1); set_pixel(d, 3, 0, 2, 0)

    set_pixel(d, 0, 1, 0, 0); set_pixel(d, 0, 1, 1, 1); set_pixel(d, 0, 1, 2, 0)
    set_pixel(d, 1, 1, 0, 0); set_pixel(d, 1, 1, 1, 1); set_pixel(d, 1, 1, 2, 1)
    set_pixel(d, 2, 1, 0, 0); set_pixel(d, 2, 1, 1, 0); set_pixel(d, 2, 1, 2, 1)
    set_pixel(d, 3, 1, 0, 1); set_pixel(d, 3, 1, 1, 0); set_pixel(d, 3, 1, 2, 1)

    # Test images are same
    TEST(same_image(d, gt, EPS) == 1,
         "test_set_pixel", "same_image(d, gt, EPS)", "src_py/test_hw1.py", 0)

    free_image(gt)
    free_image(d)


def test_grayscale():
    im = load_image("data/colorbar.png")
    gray = rgb_to_grayscale(im)
    gt = load_image("figs/gray.png")

    TEST(same_image(gray, gt, EPS) == 1,
         "test_grayscale", "same_image(gray, gt, EPS)", "src_py/test_hw1.py", 0)

    free_image(im)
    free_image(gray)
    free_image(gt)


def test_copy():
    gt = load_image("data/dog.jpg")
    c = copy_image(gt)

    TEST(same_image(c, gt, EPS) == 1,
         "test_copy", "same_image(c, gt, EPS)", "src_py/test_hw1.py", 0)

    free_image(gt)
    free_image(c)


def test_clamp():
    im = load_image("data/dog.jpg")
    c = copy_image(im)

    set_pixel(im, 10, 5, 0, -1)
    set_pixel(im, 15, 15, 1, 1.001)
    set_pixel(im, 130, 105, 2, -0.01)
    set_pixel(im, im.w - 1, im.h - 1, im.c - 1, -0.01)

    set_pixel(c, 10, 5, 0, 0)
    set_pixel(c, 15, 15, 1, 1)
    set_pixel(c, 130, 105, 2, 0)
    set_pixel(c, im.w - 1, im.h - 1, im.c - 1, 0)

    clamp_image(im)

    TEST(same_image(c, im, EPS) == 1,
         "test_clamp", "same_image(c, im, EPS)", "src_py/test_hw1.py", 0)

    free_image(im)
    free_image(c)


def test_shift():
    im = load_image("data/dog.jpg")
    c = copy_image(im)

    shift_image(c, 1, 0.1)

    wh = im.w * im.h
    c_flat = c.data.reshape(-1)
    im_flat = im.data.reshape(-1)

    TEST(within_eps(float(c_flat[0]), float(im_flat[0]), EPS),
         "test_shift", "within_eps(c.data[0], im.data[0], EPS)", "src_py/test_hw1.py", 0)
    TEST(within_eps(float(c_flat[wh + 13]), float(im_flat[wh + 13]) + 0.1, EPS),
         "test_shift", "within_eps(c.data[im.w*im.h + 13], im.data[im.w*im.h+13] + .1, EPS)", "src_py/test_hw1.py", 0)
    TEST(within_eps(float(c_flat[2 * wh + 72]), float(im_flat[2 * wh + 72]), EPS),
         "test_shift", "within_eps(c.data[2*im.w*im.h + 72], im.data[2*im.w*im.h+72], EPS)", "src_py/test_hw1.py", 0)
    TEST(within_eps(float(c_flat[wh + 47]), float(im_flat[wh + 47]) + 0.1, EPS),
         "test_shift", "within_eps(c.data[im.w*im.h + 47], im.data[im.w*im.h+47] + .1, EPS)", "src_py/test_hw1.py", 0)

    free_image(im)
    free_image(c)


def test_rgb_to_hsv():
    im = load_image("data/dog.jpg")
    rgb_to_hsv(im)
    hsv = load_image(_gt("figs/dog.hsv.png"))

    TEST(same_image(im, hsv, EPS) == 1,
         "test_rgb_to_hsv", "same_image(im, hsv, EPS)", "src_py/test_hw1.py", 0)

    free_image(im)
    free_image(hsv)


def test_hsv_to_rgb():
    im = load_image("data/dog.jpg")
    c = copy_image(im)

    rgb_to_hsv(im)
    hsv_to_rgb(im)

    TEST(same_image(im, c, EPS) == 1,
         "test_hsv_to_rgb", "same_image(im, c, EPS)", "src_py/test_hw1.py", 0)

    free_image(im)
    free_image(c)


def test_hw1():
    # EXACT order from C
    test_get_pixel()
    test_set_pixel()
    test_copy()
    test_shift()
    test_clamp()
    test_grayscale()
    test_rgb_to_hsv()
    test_hsv_to_rgb()

    print(f"{tests_total} tests, {tests_total - tests_fail} passed, {tests_fail} failed")


if __name__ == "__main__":
    test_hw1()