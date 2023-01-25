from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

from keras.models import load_model

## Source code:
# https://www.youtube.com/watch?v=qOXDoYUgNlU

model = load_model('mnist.h5')
pathImage = "sudo2.png"
heightImage = 500
widthImage = 500


image = cv2.imread('Digits.png')
image = cv2.resize(image, (widthImage, heightImage))
imageBlank = (np.zeros(heightImage, widthImage, 3), np.uint8)
imageThreshol = preProcess(image)

