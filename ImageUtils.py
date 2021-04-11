import cv2
import imutils
import numpy as np
import skimage
from scipy import ndimage
from skimage import color


def preprocess(input_image):
    img = skimage.img_as_float(input_image)
    img_dim = img.shape[2]

    if img_dim == 4:
        img = color.rgba2rgb(img)
        img_dim = 3
    if img_dim == 3:
        img_hsv = color.rgb2hsv(img)
        return img_hsv[:, :, 2]
    if img_dim > 1:
        img_hsv = color.rgb2hsv(img)
        # img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return img_hsv[:, :, 2]
    else:
        return img


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


def strech(img):
    img_cpy = np.copy(img)
    max_val = np.max(img_cpy)
    min_val = np.maximum([[-1 for i in range(img.shape[1])] for j in range(img.shape[0])], np.min(img_cpy))

    img_cpy = np.maximum(img, min_val)
    img_cpy = img_cpy - min_val
    img_cpy = img_cpy / max_val
    return img_cpy


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def rotate_img(img, theta):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
