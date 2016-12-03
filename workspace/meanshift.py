import cv2
import numpy as np

img = cv2.imread('input.jpg')
newImage = img
cv2.pyrMeanShiftFiltering(img, 10, 35, newImage)

cv2.imshow('image', newImage)
cv2.waitKey()