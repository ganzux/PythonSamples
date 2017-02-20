import cv2
import numpy as np
from matplotlib import pylab    
from matplotlib import pyplot as plt

depth = np.uint8
rows = 100
cols = 200
channel = 3

blank_image = np.zeros((rows, cols, channel), depth)
cv2.imshow("Black Windows", blank_image)
cv2.waitKey()

m = cv2.imread("avatar.jpg")    # Read JPG image
cv2.imshow("AvatarWindow", m)   # Show the Window
cv2.waitKey()                   # Wait for a key
cv2.destroyAllWindows()         # Close all windows

pylab.ion()
plt.imshow(cv2.cvtColor(m, cv2.COLOR_BGR2RGB)) # From BGR to RGB. BGR (intern OpenCV format by default for color images)
