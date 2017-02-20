import cv2
import numpy as np

depth = np.uint8
rows = 100
cols = 200
channels = 3
blank_image = np.zeros((rows,cols,channels), depth)
cv2.imshow("BlackWindow", blank_image)
cv2.waitKey()

vec = np.array([1., 2., 1.])                  # row Vector, 1 x 3 dimension 
mat = np.array([[1., 2., 3.], [4., 5., 6.]])  # 2 x 3 Matrix

m_0 = np.zeros( (3, 4) )                    # 3 x 4 Matrix fill with zeros (float64) 
m_1 = np.ones( (2, 3, 4), dtype=np.int16 )  # 2 x 3 x 4 Matrix fill with ones (16 bit int)
m_2 = np.random.rand(4,3)                   # 4 x 3 Matrix fill with Random numbers (float64)


roi = m_2[0:3, 1:3] # SubMatrix rows 0 - 2 (both inclusive) until 1 - 2 columns (inclusive)
roi = -1            # assign -1 to all the elements
m_5 = m_2[1, :]     # second row of the matrix
m_6 = m_2[: ,2]     # third column of the Matrix


a = np.zeros((10, 20))    # 10 x 20 zero Matrix
b = a                     # b is a pointer to a
a[0, 0] = 23              # Assign value
change_b = b[0, 0] == 23  # true because b is a pointer

b = a.copy()            # Now b is a copy, but they don't share memory
a[0, 0] = 2             # Assign value
change_b = b[0, 0] == 2 # false, a and b are not the same
change = (b == a).all() # false, a and b are not the same

# Some numpy aclarations
# shape   tupla with the dimensions: rows, columns, color planes
# ndim    shape dimensions: 2 for grey, 3 for color
# dtype   elements type (int8, float64, etc.)
# “[]”    for accesing to matrix elements m[0, 2] = 3
# ":"     accesing to row or column m[:, 0] = 0
# i:j:k   slicing fron i to j (j not included) by k steps.
#         m[0:2, 1:5:2] subMatrix with the rows 0-1 and columns 1 and 3
#

# Matrix Types and Images
# dtype = np.uint8 and ndim = 2, for images on grey scale [0 - 255] or bitonal p{0, 255}
i_8 = np.array([[1, 2, 3] ,[4, 5, 6]], dtype = np.uint8)

# dtype = np.uint8 and ndim = 3, for color images [0-255] and 3 color channels (24 bits per pixel)
# In OpenCV by default all the images are BGR
i_24 = np.array([[[1, 2, 3],  [4,   5, 6]], 
                [[7, 8, 9],  [10, 11, 12]], 
                [[13, 14, 15], [16, 17, 18]]], dtype = np.uint8)
# dtype = np.int32 and ndim = 2, for 32 bits integer matrices (int64 for 64 bits):
m_int = np.array([[1, 2, 3], [4, 5, 6]], dtype = np.int32)
# dtype = np.float32 and ndim = 2, for 32 bits float matrices (or float16, float64, float128):
m_float = np.array([[1, 2, 3], [4, 5, 6]], dtype = np.float32) 

# The operators  +, -, * and / are overriden for the matrices with the same dimension and type
# Also with |, &
# ~, >, <, >= y <= 
# Get an image with true (255) and false (0) with the pixels which value is greater than 200
grey_image = cv2.imread("avatar.jpg")
high_pixels = (grey_image > 200)  # true - false mask
high_pixels = high_pixels * 255   # mask in range [0, 255]
cv2.imshow("BlackWindow", high_pixels)
cv2.waitKey()

# DRAWING!
# line.
# rectangle.
# circle.
# ellipse.
# polylines.
# fillpoly.
# putText.

# dynamic typed
def fact(value : int) -> int :
  if value == 0:
    return 1
  return value * fact(value - 1)
