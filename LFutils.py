import numpy as np
import imutils
import scipy.ndimage as ndimage
from scipy.signal import convolve2d
from cv2 import resize as resize
import skimage


def to_rad(deg):
    return (deg * np.pi) / 180


def to_degrees(rad):
    return (rad / np.pi) * 180


def apply_img_filter(img, f, mode='correlate'):
    if mode == 'correlate':
        return ndimage.correlate(img, f, mode='nearest').transpose()
    elif mode == 'conv':
        return ndimage.convolve(img, f, mode='nearest')


def fspecial_motion_blur(length, angle):
    # First generate a horizontal line across the middle
    shape = (length, length)
    f = np.zeros(shape)
    f[length // 2 + 1, 1:length] = 1

    # Then rotate to specified angle
    f = imutils.rotate_bound(f, angle)
    f = f / sum(f[:])
    return f


def get_gabor_filter(angle=0, length=25, sig=8, gamma=1, lmd=12, psi=0, is_radian=True):
    # get half size
    d = length // 2
    
    if (not is_radian):
      # degree -> radian
      theta = to_rad(angle)
    else:
      theta = angle

    # prepare kernel
    gabor = np.zeros((length, length), dtype=np.float32)
    l_norm = np.zeros((length, length))

    # each value
    for y in range(length):
        for x in range(length):
            # distance from center
            px = x - d
            py = y - d

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + gamma ** 2 * _y ** 2) / (2 * sig ** 2)) * np.cos(2 * np.pi * _x / lmd + psi)
            l_norm[y][x] = np.cos((2 * np.pi / lmd) * _x)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))
    gabor = resize(gabor, (np.sqrt(length) // 1, np.sqrt(length) // 1))
    l_norm = resize(l_norm, (length // 5, length // 5))

    return gabor, l_norm

def line_filling_sc(img, theta, fac, c=0.05):
    """theta must be degrees (NOT radiant!)"""
    img_NR = np.power(img, 2) / (np.power(img, 2) + c ** 2)
    motion_blur_ker = fspecial_motion_blur(int(fac), 90-theta)
    blurred_im = 2 * apply_img_filter(img_NR, motion_blur_ker)
    motion_blur_ker = fspecial_motion_blur(int(np.round(fac/2)), 90-theta)
    k = max(fac/4, 1)
    return k * (blurred_im + apply_img_filter(img_NR, motion_blur_ker)).T , img_NR

# bresenham function is the accepted answer of SO's post https://stackoverflow.com/questions/23930274/list-of-coordinates-between-irregular-points-in-python
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))

    return points


def strel_line(length, degrees):
    if length >= 1:
        theta = degrees * np.pi / 180
        x = round((length - 1) / 2 * np.cos(theta))
        y = -round((length - 1) / 2 * np.sin(theta))
        points = bresenham(-x, -y, x, y)
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        n_rows = int(2 * max([abs(point_y) for point_y in points_y]) + 1)
        n_columns = int(2 * max([abs(point_x) for point_x in points_x]) + 1)
        strel = np.zeros((n_rows, n_columns))
        rows = ([point_y + max([abs(point_y) for point_y in points_y]) for point_y in points_y])
        columns = ([point_x + max([abs(point_x) for point_x in points_x]) for point_x in points_x])
        idx = []
        for x in zip(rows, columns):
            idx.append(np.ravel_multi_index((int(x[0]), int(x[1])), (n_rows, n_columns)))
        strel.reshape(-1)[idx] = 1

    return skimage.img_as_ubyte(strel).T

def strech(img):
    img_cpy = np.copy(img)
    max_val = np.max(img_cpy)
    min_val = np.maximum([[-1 for i in range(img.shape[1])] for j in range(img.shape[0])], np.min(img_cpy))

    img_cpy = np.maximum(img, min_val)
    img_cpy = img_cpy - min_val
    img_cpy = img_cpy / max_val
    return img_cpy