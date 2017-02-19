import cv2
import numpy as np
depth = np.uint8
rows = 100
cols = 200
channel = 3

blank_image = np.zeros((rows, cols, channel), depth)
cv2.imshow("Black Windows", blank_image)
cv2.waitKey()