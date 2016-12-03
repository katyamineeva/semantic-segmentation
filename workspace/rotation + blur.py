import cv2
import numpy as np

img = cv2.imread('input.jpg')
num_rows, num_cols = img.shape[:2]

gausBlur = cv2.GaussianBlur(img, (5, 5), 0)
medBlur = cv2.medianBlur(gausBlur, 5)


rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 15, 1)

img_rotation = cv2.warpAffine(medBlur, rotation_matrix, (num_cols, num_rows))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey()