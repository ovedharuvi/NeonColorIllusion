# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import skimage

from FalseContours import detect_false_contour
from Params import *
import matplotlib.pylab as plt
from skimage import io, color

img = io.imread(image_path)
orig = img
# img = img[:, :, :-1]
print('Image shape:{}, Image data type:{}'.format(img.shape, img.dtype))
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X & Y axis
plt.show()

guesses_sum, threshold_guesses_sum = detect_false_contour(img)

plt.imshow(threshold_guesses_sum, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X & Y axis
plt.show()

img = color.rgb2gray(img)
plt.imshow(skimage.img_as_ubyte(img) - threshold_guesses_sum, cmap='gray')
plt.show()
