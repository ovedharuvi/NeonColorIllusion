# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ImageUtils import *
from FalseContours import detect_false_contour
from Params import *
import matplotlib.pylab as plt
from skimage import io, color
import logging

logging.basicConfig(level=logging.INFO)
img = io.imread(image_path)
img = init_img(img)
orig = img
logging.info('Image shape:{}, Image data type:{}'.format(img.shape, img.dtype))
plt.figure(figsize=(30,15))
plt.subplot(131)
plt.imshow(img , cmap='gray')
plt.title("Original Image")

img = preprocess(img)
guesses_sum, threshold_guesses_sum = detect_false_contour(img, orig)
guesses_sum = post_process(guesses_sum)
threshold_guesses_sum = post_process(threshold_guesses_sum)

plt.subplot(132)
plt.imshow(threshold_guesses_sum, cmap='gray')
plt.title("Guesses with threshold")
#
# plt.imshow(guesses_sum, cmap='gray')
# plt.title("Guesses without threshold")
# plt.xticks([]), plt.yticks([])  # to hide tick values on X & Y axis
# plt.show()

# if orig.shape[2] >= 3:
#     threshold_guesses_sum = skimage.img_as_ubyte(skimage.color.gray2rgb(threshold_guesses_sum))
orig_final = skimage.img_as_ubyte(orig)
final = draw_contours(orig_final, threshold_guesses_sum, IS_GRAY)

plt.subplot(133)
plt.title("False Contour Algorithm Result")
plt.imshow(final, cmap='gray')
plt.show()

plt.imshow(final, cmap='gray')
plt.show()
