import numpy as np
import imutils
import scipy.ndimage as ndimage


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
    f = np.zeros(length)
    f[np.floor(length / 2) + 1, 1:length] = 1

    # Then rotate to specified angle
    f = imutils.rotate_bound(f, angle)
    f = f / sum(f[:])
    return f


def get_gabor_filter(theta, lmd=12, sig=8, x_b=13, y_b=13, length=25):
    """theta must be radiant (NOT degrees!)"""
    xs = ys = [k for k in range(length)]
    l = np.zeros((length, length))
    l_norm = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            exp = np.exp(np.e, -((xs[i]-x_b) ^ 2 + (ys[j]-y_b) ^ 2)/(sig ^ 2))
            k = (xs[i]-x_b) * np.cos(theta) + (ys[i]-y_b) * np.sin(theta)
            l[j][i] = np.cos((2*np.pi / lmd) * k)
            l_norm[j][i] = l[j][i]
            l[j][i] *= exp
    return l, l_norm


def line_filling_sc(img, theta, fac, c=0.05):
    """theta must be degrees (NOT radiant!)"""
    img_NR = np.power(img, 2) / (np.power(img, 2) + c ^ 2)
    motion_blur_ker = fspecial_motion_blur(fac, 90-theta)
    blurred_im = 2 * apply_img_filter(img_NR, motion_blur_ker)
    motion_blur_ker = fspecial_motion_blur(np.round(fac/2), 90-theta)
    k = max(fac/4, 1)
    return k * (blurred_im + apply_img_filter(img_NR, motion_blur_ker)), img_NR

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

    return strel
