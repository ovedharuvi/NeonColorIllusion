import cv2
import imutils
import numpy as np
import skimage
import skimage.transform
from scipy import ndimage
from skimage import color

from Params import *


def preprocess(input_image):
    img = skimage.img_as_float(input_image)
    img_dim = img.shape[2]

    if img_dim == 4:
        img = color.rgba2rgb(img)
        img_dim = 3
    if img_dim == 3:
        img_hsv = color.rgb2hsv(img)
        img = img_hsv[:, :, 2]
    elif img_dim > 1:
        img_hsv = color.rgb2hsv(img)
        img = img_hsv[:, :, 2]
    padded_img = np.pad(img, (PAD_SIZE, PAD_SIZE), constant_values=1.0)
    return padded_img


def draw_contours(img, contours, is_gray):
    if is_gray:
        img = skimage.color.rgb2gray(img)
    if is_gray or len(img.shape) == 2 or img.shape[2] == 1:
        return img - contours
    contours = skimage.img_as_ubyte(normalize(contours))
    contours, hier = cv2.findContours(contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    real_contours = []
    for c in contours:
        if cv2.contourArea(c) > 2 * np.sqrt(img.shape[0] * img.shape[1]):
            x, y, w, h = cv2.boundingRect(c)
            cropped_area = img[y:y + h, x:x + w, :]
            hue = find_contour_hue(cropped_area)
            real_contours.append((c, hue))
    result = img
    white = skimage.img_as_ubyte(np.ones(img.shape))
    for cnt, hue in real_contours:
        hue = (int(hue[0]), int(hue[1]), int(hue[2]))
        cv2.drawContours(image=white, contours=[cnt], contourIdx=-1,
                         color=hue,
                         thickness=2, lineType=cv2.LINE_AA)

        # hue = colorsys.rgb_to_hls(hue[0], hue[1], hue[2])
        # hue = (hue[0], 250, hue[2])
        # hue = colorsys.hls_to_rgb(hue[0], hue[1], hue[2])
        # cv2.fillConvexPoly(white, cnt, color=(int(hue[0]), int(hue[1]), int(hue[2])))
    return white, real_contours


def find_contour_hue(img):
    # yiq_img = color.rgb2yiq(img)
    # colorfulness_img = yiq_img[:, :, 2]**2 + yiq_img[:, :, 0]**2
    # pix_argmax = np.argmax(colorfulness_img)
    # max_color_index = np.unravel_index(pix_argmax, colorfulness_img.shape)
    # hue = img[max_color_index[0]][max_color_index[1]]

    (R, G, B) = cv2.split(img.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    colorfulness_img = rg ** 2 + yb ** 2
    pix_argmax = np.argmax(colorfulness_img)
    max_color_index = np.unravel_index(pix_argmax, colorfulness_img.shape)
    hue = img[max_color_index[1]][max_color_index[0]]
    return hue


def bounding_box(cnt):
    rect = cv2.boundingRect(cnt)
    box = cv2.boxPoints(rect)
    tl = tuple([int(box[0][0]), int(box[0][1])])
    bl = tuple([int(box[1][0]), int(box[1][1])])
    br = tuple([int(box[2][0]), int(box[2][1])])
    tr = tuple([int(box[3][0]), int(box[3][1])])
    top = max(tl, tr)
    bottom = min(bl, br)
    right = max(br, tr)
    left = min(bl, tl)
    return top, bottom, right, left


def post_process(img):
    img = normalize(img)
    if len(img.shape) >= 3:
        result = img[PAD_SIZE:img.shape[0] - PAD_SIZE, PAD_SIZE:img.shape[1] - PAD_SIZE, :]
    else:
        result = img[PAD_SIZE:img.shape[0] - PAD_SIZE, PAD_SIZE:img.shape[1] - PAD_SIZE]
    return result


def to_rad(deg):
    return (deg * np.pi) / 180


def init_img(img):
    img_dim = img.shape[2]
    result = img
    if img_dim == 4:
        result = skimage.img_as_ubyte(color.rgba2rgb(img))
    return result


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
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
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
