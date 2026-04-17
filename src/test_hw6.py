import os
import inspect
import numpy as np

import uwimg
from uwimg import Image, load_image, make_image, free_image, save_png
from src.hw1.process_image import get_pixel, set_pixel
from src.hw6.flow_image import (
	make_integral_image,
	box_filter_image,
	time_structure_matrix,
	velocity_image,
)

EPS = 0.005
PRINT_PASSES = True


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
_current_test_name = "unknown"


def within_eps(a: float, b: float, eps: float) -> bool:
	return (a - eps) < b and b < (a + eps)


def TEST(cond: bool, expr: str) -> None:
	global tests_total, tests_fail
	tests_total += 1
	if not cond:
		tests_fail += 1
		fr = inspect.currentframe().f_back
		rel = os.path.relpath(fr.f_code.co_filename)
		line = fr.f_lineno
		print(f"failed: [{_current_test_name}] testing [{expr}] in {rel}, line {line}")
	else:
		if PRINT_PASSES:
			print(f"passed: [{_current_test_name}] testing [{expr}]")


def same_image(a: Image, b: Image, eps: float) -> bool:
	if a.w != b.w or a.h != b.h or a.c != b.c:
		return False

	ad = a.data.reshape(-1)
	bd = b.data.reshape(-1)
	for i in range(a.w * a.h * a.c):
		thresh = (abs(float(bd[i])) + abs(float(ad[i]))) * eps / 2.0
		if thresh > eps:
			eps = thresh
		if not within_eps(float(ad[i]), float(bd[i]), eps):
			print(f"The value should be {float(bd[i])}, but it is {float(ad[i])}! ")
			return False
	return True


def load_image_binary(path: str) -> Image:
	with open(path, "rb") as f:
		shape = np.fromfile(f, dtype=np.int32, count=3)
		if shape.size != 3:
			raise ValueError(f"Invalid binary image header: {path}")
		w, h, c = int(shape[0]), int(shape[1]), int(shape[2])
		n = w * h * c
		data = np.fromfile(f, dtype=np.float32, count=n)
		if data.size != n:
			raise ValueError(f"Invalid binary image payload: {path}")

	im = make_image(w, h, c)
	im.data[...] = data.reshape((c, h, w))
	return im


def save_image_binary(im: Image, path: str) -> None:
	with open(path, "wb") as f:
		np.array([im.w, im.h, im.c], dtype=np.int32).tofile(f)
		im.data.reshape(-1).astype(np.float32).tofile(f)


def avg_diff(a: Image, b: Image) -> float:
	diff = 0.0
	ad = a.data.reshape(-1)
	bd = b.data.reshape(-1)
	for i in range(a.w * a.h * a.c):
		diff += float(bd[i]) - float(ad[i])
	return diff / (a.w * a.h * a.c)


def center_crop(im: Image) -> Image:
	c = make_image(im.w // 2, im.h // 2, im.c)
	for k in range(im.c):
		for j in range(im.h // 2):
			for i in range(im.w // 2):
				set_pixel(c, i, j, k, get_pixel(im, i + im.w // 4, j + im.h // 4, k))
	return c


def make_hw6_tests() -> None:
	dots = load_image("data/dots.png")
	intdot = make_integral_image(dots)
	save_image_binary(intdot, "data/dotsintegral.bin")

	dogbw = load_image("data/dogbw.png")
	intdog = make_integral_image(dogbw)
	save_image_binary(intdog, "data/dogintegral.bin")

	dog = load_image("data/dog.jpg")
	smooth = box_filter_image(dog, 15)
	save_png(smooth, "data/dogbox")

	smooth_c = center_crop(smooth)
	save_png(smooth_c, "data/dogboxcenter")

	doga = load_image("data/dog_a_small.jpg")
	dogb = load_image("data/dog_b_small.jpg")
	structure = time_structure_matrix(dogb, doga, 15)
	save_image_binary(structure, "data/structure.bin")

	velocity = velocity_image(structure, 5)
	save_image_binary(velocity, "data/velocity.bin")


def test_integral_image() -> None:
	global _current_test_name
	_current_test_name = "test_integral_image"

	dots = load_image("data/dots.png")
	intdot = make_integral_image(dots)
	intdot_t = load_image_binary("data/dotsintegral.bin")
	TEST(same_image(intdot, intdot_t, EPS), "same_image(intdot, intdot_t, EPS)")

	dog = load_image("data/dogbw.png")
	intdog = make_integral_image(dog)
	intdog_t = load_image_binary("data/dogintegral.bin")
	TEST(same_image(intdog, intdog_t, 0.6), "same_image(intdog, intdog_t, .6)")

	free_image(dots)
	free_image(intdot)
	free_image(intdot_t)
	free_image(dog)
	free_image(intdog)
	free_image(intdog_t)


def test_exact_box_filter_image() -> None:
	global _current_test_name
	_current_test_name = "test_exact_box_filter_image"

	dog = load_image("data/dog.jpg")
	smooth = box_filter_image(dog, 15)
	smooth_t = load_image(_gt("data/dogbox.png"))
	TEST(same_image(smooth, smooth_t, EPS * 2), "same_image(smooth, smooth_t, EPS*2)")

	free_image(dog)
	free_image(smooth)
	free_image(smooth_t)


def test_good_enough_box_filter_image() -> None:
	global _current_test_name
	_current_test_name = "test_good_enough_box_filter_image"

	dog = load_image("data/dog.jpg")
	smooth = box_filter_image(dog, 15)
	smooth_c = center_crop(smooth)
	smooth_t = load_image(_gt("data/dogboxcenter.png"))
	print(f"avg origin difference test: {avg_diff(smooth_c, center_crop(dog))}")
	print(f"avg smooth difference test: {avg_diff(smooth_c, smooth_t)}")
	TEST(same_image(smooth_c, smooth_t, EPS * 2), "same_image(smooth_c, smooth_t, EPS*2)")

	free_image(dog)
	free_image(smooth)
	free_image(smooth_c)
	free_image(smooth_t)


def test_structure_image() -> None:
	global _current_test_name
	_current_test_name = "test_structure_image"

	doga = load_image("data/dog_a_small.jpg")
	dogb = load_image("data/dog_b_small.jpg")
	structure = time_structure_matrix(dogb, doga, 15)
	structure_t = load_image_binary(_gt("data/structure.bin"))
	TEST(
		same_image(center_crop(structure), center_crop(structure_t), EPS),
		"same_image(center_crop(structure), center_crop(structure_t), EPS)",
	)

	free_image(doga)
	free_image(dogb)
	free_image(structure)
	free_image(structure_t)


def test_velocity_image() -> None:
	global _current_test_name
	_current_test_name = "test_velocity_image"

	structure = load_image_binary(_gt("data/structure.bin"))
	velocity = velocity_image(structure, 5)
	velocity_t = load_image_binary(_gt("data/velocity.bin"))
	TEST(same_image(velocity, velocity_t, EPS), "same_image(velocity, velocity_t, EPS)")

	free_image(structure)
	free_image(velocity)
	free_image(velocity_t)


def test_hw6() -> None:
	# make_hw6_tests()
	test_integral_image()
	test_exact_box_filter_image()
	test_good_enough_box_filter_image()
	test_structure_image()
	test_velocity_image()
	print(f"{tests_total} tests, {tests_total - tests_fail} passed, {tests_fail} failed")


if __name__ == "__main__":
	test_hw6()
