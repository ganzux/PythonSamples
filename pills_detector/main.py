import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import itertools

########################################################################################################################
#                                                   VARIABLES
########################################################################################################################
# Empty Array for the data
from astropy.config.paths import set_temp_config
from scipy.fftpack.tests.test_basic import direct_dft

train = np.empty((0,18), dtype=np.float32)

# Empty Array for the label class of every image
train_labels = np.empty((0,1), dtype=np.float32)

# All labels for the 20 classes [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
labels = [i for i in range(21) if i > 0]


# Dictionary with the data of the colors, so:
# name, low colors, high colors,
colors = {"red2" :  [[144, 71, 48], [144, 24, 98]],
          "white" : [[0, 0, 0], [100, 100, 255]],
          "orange": [[0,150,150], [30, 255, 255]],
          "pink":   [[150, 80, 20], [190, 255, 255]],
          "yellow": [[25, 80, 80], [34, 255, 255]],
          "green":  [[50, 80, 60], [85, 255, 255]],
          "blue":   [[90, 80, 60], [120, 255, 255]],
          "red":    [[0, 90,  60], [20, 255, 255]]}

# Folders with the images and the different classes and the colors it has
data = {"1" : ["images/Train/clase1/", colors["red"]],
        "2" : ["images/Train/clase2/", colors["white"]],
        "3" : ["images/Train/clase3/", colors["pink"], colors["blue"]],
        "4" : ["images/Train/clase4/", colors["white"], colors["orange"]],
        "5" : ["images/Train/clase5/", colors["red"], colors["white"]],
        "6" : ["images/Train/clase6/", colors["white"]],
        "7" : ["images/Train/clase7/", colors["blue"]],
        "8" : ["images/Train/clase8/", colors["pink"]],
        "9" : ["images/Train/clase9/", colors["blue"]],
        "10": ["images/Train/clase10/",colors["green"], colors["yellow"]],
        "11": ["images/Train/clase11/",colors["green"], colors["white"]],
        "12": ["images/Train/clase12/",colors["orange"]],
        "13": ["images/Train/clase13/",colors["red"], colors["white"]],
        "14": ["images/Train/clase14/",colors["yellow"]],
        "15": ["images/Train/clase15/",colors["red"], colors["white"]],
        "16": ["images/Train/clase16/",colors["white"]],
        "17": ["images/Train/clase17/",colors["white"]],
        "18": ["images/Train/clase18/",colors["red"], colors["white"]],
        "19": ["images/Train/clase19/",colors["orange"], colors["white"]],
        "20": ["images/Train/clase20/",colors["white"]]}

# Folders with the test images folders
testData = {"1" : ["images/Test/clase1/"],
            "2" : ["images/Test/clase2/"],
            "3" : ["images/Test/clase3/"],
            "4" : ["images/Test/clase4/"],
            "5" : ["images/Test/clase5/"],
            "6" : ["images/Test/clase6/"],
            "7" : ["images/Test/clase7/"],
            "8" : ["images/Test/clase8/"],
            "9" : ["images/Test/clase9/"],
            "10": ["images/Test/clase10/"],
            "11": ["images/Test/clase11/"],
            "12": ["images/Test/clase12/"],
            "13": ["images/Test/clase13/"],
            "14": ["images/Test/clase14/"],
            "15": ["images/Test/clase15/"],
            "16": ["images/Test/clase16/"],
            "17": ["images/Test/clase17/"],
            "18": ["images/Test/clase18/"],
            "19": ["images/Test/clase19/"],
            "20": ["images/Test/clase20/"]}

test = False                # Show the windows with the images for the training phase
wait_key_time = 100          # Time to wait between Windows
save_files = False           # If we want to generate and save the file with the data
read_files = False           # If we want to read the file from the disk
file_name = "knn_data.npz"  # File name
training = True             # Do training Phase

########################################################################################################################
#                                                   FUNCTIONS
########################################################################################################################

print(np.__version__)
np.show_config()

def saveData():
    print("Saving data into " + file_name)
    if save_files:
        np.savez(file_name, train=train, train_labels=train_labels)
    print("Saved data into " + file_name + " done!")

def loadData():
    global train
    global train_labels

    print("Reading data from " + file_name)
    with np.load(file_name) as data:
        print (data.files)
        train = data['train']
        train_labels = data['train_labels']

        #train = train.reshape((train.size, 1))

    print("Data from " + file_name + " loaded!")

def showImage(img, text, x, y):
    if test:
        cv2.imshow(text, img)
        cv2.moveWindow(text, x, y)

# Create and Apply mask for an image
def definer(hsv, imagen, l, h):
    low = np.array(l)
    high = np.array(h)

    # Create the mask with the low and high colors based in hsv
    mask = np.zeros(hsv.shape[:2], np.uint8)
    mask = cv2.inRange(hsv, low, high)
    # Remove small dots
    mask = cv2.medianBlur(mask, 7)
    # Combine the binary image with the color ofthe original one
    new_image = cv2.bitwise_and(imagen, imagen, mask=mask)
    # Get the contours of the mask
    img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return new_image, img, contours, hierarchy

def loadTraining():
    global train
    global train_labels

    # Instantiate the kNN algorithm
    knn = cv2.ml.KNearest_create()
    # Then, we pass the trainData and responses to train the kNN

    print(type(train_labels))
    print(np.shape(train_labels))
    print(type(train))
    print(np.shape(train))
    print(type(train[1][1]))

    train_labels = train_labels.astype(np.float32)
    train = train.astype(np.float32)

    print(type(train[1][1]))

    knn.train(np.float32(train), cv2.ml.ROW_SAMPLE, np.float32(train_labels))

    return knn

def loadCentroids(thresh):
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)

    try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    except ZeroDivisionError:
        cx = 0
        cy = 0

    return cx, cy

# Function to read all the training data
def readTraining():

    global train
    global train_labels

    # reading all the training folders
    for key in data:
        print (key, data[key])

        class_name = key
        directory = data[key][0]
        first_color = data[key][1]
        second_color = None
        if len(data[key]) > 2:
            second_color = data[key][2]

        print ("Class " + class_name + " -> " + directory)

        # go through all the JPG files
        files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.lower().endswith(".jpg")]
        for f in files:

            print("Image: " + f + " init")
            image = cv2.imread(directory + f)

            #cx, cy, x, y, w, h = getImageInfo(image, first_color, second_color, class_name)
            cx, cy, x, y, w, h, lc11, lc12, lc13, hc11, hc12, hc13, lc21, lc22, lc23, hc21, hc22, hc23  = getImageInfo(
                image, first_color, second_color, class_name)

            print("Adding Label ..." + class_name)
            train_labels = np.append(train_labels,
                                     np.array([[float(class_name)]]),
                                     axis=0)

            # Add the image information into the Array with the data
            print("Adding data to train...")
            train = np.append(train,
                              np.array([[cx, cy, x, y, w, h, lc11, lc12, lc13, hc11, hc12, hc13, lc21, lc22, lc23, hc21, hc22, hc23]]),
                              axis=0)

            print("Image: " + f + " end")

def testAlgorithm(knn):
    # reading all the training folders
    index = 0
    ok = 0
    ko = 0

    for key in testData:
        print(key, testData[key])

        class_name = key
        directory = testData[key][0]

        print("Class " + class_name + " -> " + directory)

        # go through all the JPG files
        files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.lower().endswith(".jpg")]
        for f in files:

            index += 1

            print("Test Image: " + f + " init")
            image = cv2.imread(directory + f)

            #cx, cy, x, y, w, h = getImageInfo(image, None, None, "Testing")
            cx, cy, x, y, w, h, lc11, lc12, lc13, hc11, hc12, hc13, lc21, lc22, lc23, hc21, hc22, hc23 = getImageInfo(
                image, None, None, "Testing")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Create an Adaptative ThresHold for the image, so we can get the border
            th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY, 11, 2)

            roi = th3[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)

            my_array = np.empty((0, 18), dtype=np.float32)
            my_array = np.append(my_array,
                                 np.array([[cx, cy, x, y, w, h, lc11, lc12, lc13, hc11, hc12, hc13, lc21, lc22, lc23, hc21, hc22, hc23]]),
                                 axis=0)

            ret, result, neighbours, dist = knn.findNearest(np.float32(my_array), 1)

            print(ret)
            print(result)
            print(neighbours)
            print(dist)

            if str(int(result[0][0])) == class_name:
                ok += 1
            else:
                ko += 1

            if test:
                cv2.waitKey(wait_key_time)

    print("Total: {}, OK: {}, KO: {}".format(index, ok, ko))

    cv2.destroyAllWindows()  # Close all windows

def getColorInfo(hsv, image, th3,j,i,w,h):
    # loop over the boundaries
    maxColorsInMask = 0
    color = next(iter(colors.values()))
    #rect = cv2.rectangle(image, (x,y),(x+w,y+h), (0,0,255), 3)
    rect = image[i:i + h, j:j + w]
    median = rect.mean()
    for key in colors:
        lower_upper = colors[key]
        # create NumPy arrays from the boundaries
        lower = np.array(lower_upper[0], dtype="uint8")
        upper = np.array(lower_upper[1], dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(hsv, hsv, mask=mask)

        pixelInMask = cv2.countNonZero(mask)
        if (pixelInMask > maxColorsInMask):
            maxColorsInMask = pixelInMask
            color = key

    return colors[color]

def getColorsVariables(color):
    lc1 = 0
    lc2 = 0
    lc3 = 0
    hc1 = 0
    hc2 = 0
    hc3 = 0
    if color != None:
        lc1 = color[0][0]
        lc2 = color[0][1]
        lc3 = color[0][2]
        hc1 = color[1][0]
        hc2 = color[1][1]
        hc3 = color[1][2]
    return lc1, lc2, lc3, hc1, hc2, hc3

def getImageInfo(image, first_color, second_color, class_name):
    print("getImageInfo: " + class_name + " init")

    # Gaussian filter to the image
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # Image with the color composition HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # image in gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create an Adaptative ThresHold for the image, so we can get the border
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 2)

    x, y, w, h = cv2.boundingRect(th3)
    cx, cy = loadCentroids(th3)

    if first_color == None:
        first_color = getColorInfo(hsv, image, th3, x, y, w, h)

    if second_color == None:
        second_color = first_color

    lc11, lc12, lc13, hc11, hc12, hc13 = getColorsVariables(first_color)
    lc21, lc22, lc23, hc21, hc22, hc23 = getColorsVariables(second_color)

    showImage(image, "Color Image, class " + class_name, 0, 0)
    showImage(gray, "Grey, class " + class_name, 0, 500)
    showImage(th3, "Adaptive Threshold, class " + class_name, 0, 1000)

    if test:
        cv2.waitKey(wait_key_time)

    cv2.destroyAllWindows()

    return cx, cy, x, y, w , h, lc11, lc12, lc13, hc11, hc12, hc13, lc21, lc22, lc23, hc21, hc22, hc23

########################################################################################################################
#                                                   MAIN PROGRAM
########################################################################################################################



if training:
    readTraining()

if save_files:
    saveData()

if read_files:
    loadData()

knn = loadTraining()

testAlgorithm(knn)
