import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

########################################################################################################################
#                                                   VARIABLES
########################################################################################################################
# Empty Array for the data TODO, total unmber of files to train
#train =  np.empty(0,300)
train =  []

# Empty Array for the label class of every image
train_labels = []

# All labels for the 20 classes [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
labels = [i for i in range(21) if i > 0]


# Dictionary with the data of the training file, so:
# directory, low colors, high colors,
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
save_files = True           # If we want to generate and save the file with the data
read_files = True           # If we want to read the file from the disk
file_name = "knn_data.npz"  # File name
training = True             # Do training Phase

########################################################################################################################
#                                                   FUNCTIONS
########################################################################################################################

def saveData():
    print("Saving data into " + file_name)
    if save_files:
        np.savez(file_name, train=train, train_labels=train_labels)
    print("Saved data into " + file_name + " done!")

def loadData():
    print("Reading data from " + file_name)
    with np.load(file_name) as data:
        print (data.files)
        train = data['train']
        train_labels = data['train_labels']
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
    mask = cv2.inRange(hsv, low, high)
    # Remove small dots
    mask = cv2.medianBlur(mask, 7)
    # Combine the binary image with the color ofthe original one
    new_image = cv2.bitwise_and(imagen, imagen, mask=mask)
    # Get the contours of the mask
    img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return new_image, img, contours, hierarchy

def loadTraining():
    if read_files:
        loadData()

    # Instantiate the kNN algorithm
    knn = cv2.ml.KNearest_create()
    # Then, we pass the trainData and responses to train the kNN
    knn.train(train, train_labels)

    return knn

# Function to read all the training data
def readTraining():

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
            print("Image: " + f)
            imagen = cv2.imread(directory + f)

            # Gaussian filter to the image
            blur = cv2.GaussianBlur(imagen, (3, 3), 0)
            # Image with the color composition HSV
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            # image in gray scale
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            # Create color mask for the image
            low = np.array(first_color[0])
            high = np.array(first_color[1])
            color_mask, img, contours, hierarchy = definer(hsv, imagen, low, high)

            # Create an Adaptative ThresHold for the image, so we can get the border
            th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY, 11, 2)

            # Calculate mask for the second color if it does exist
            if second_color != None:
                low2 = np.array(second_color[0])
                high2 = np.array(second_color[1])
                color_mask2, img2, contours2, hierarchy2 = definer(hsv, imagen, low2, high2)
                showImage(color_mask2, "Mask 2, class " + class_name, 700, 500)

            showImage(imagen, "Color Image, class " + class_name, 0, 0)
            showImage(color_mask, "Mask 1, class " + class_name, 700, 0)
            showImage(gray, "Grey, class " + class_name, 0, 500)
            showImage(th3, "Adaptive Threshold, class " + class_name, 0, 1000)

            if test:
                cv2.waitKey(wait_key_time)

            # Add the class name to the image labels Array
            train_labels.append(class_name)

            # Add the image information into the Array with the data
            if second_color != None:
                train.append([color_mask, img, contours, hierarchy, color_mask2, img2, contours2, hierarchy2])
            else:
                train.append([color_mask, img, contours, hierarchy])

            cv2.destroyAllWindows()

    if save_files:
        saveData()


def testAlgorithm(knn):
    # reading all the training folders
    for key in testData:
        print(key, data[key])

        class_name = key
        directory = data[key][0]

        print("Class " + class_name + " -> " + directory)

        # go through all the JPG files
        files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.lower().endswith(".jpg")]
        for f in files:
            print("Image: " + f)
            imagen = cv2.imread(directory + f)

            ret, result, neighbours, dist = knn.find_nearest(imagen, k=5)

            print(ret)
            print(result)
            print(neighbours)
            print(dist)

            cv2.waitKey(wait_key_time)

            """# Gaussian filter to the image
            blur = cv2.GaussianBlur(imagen, (3, 3), 0)
            # Image with the color composition HSV
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            # image in gray scale
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            # Create color mask for the image
            low = np.array(first_color[0])
            high = np.array(first_color[1])
            color_mask, img, contours, hierarchy = definer(hsv, imagen, low, high)

            # Create an Adaptative ThresHold for the image, so we can get the border
            th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY, 11, 2)


            showImage(imagen, "Color Image, class " + class_name, 0, 0)
            showImage(color_mask, "Mask 1, class " + class_name, 700, 0)
            showImage(gray, "Grey, class " + class_name, 0, 500)
            showImage(th3, "Adaptive Threshold, class " + class_name, 0, 1000)"""""

            if test:
                cv2.waitKey(wait_key_time)

    cv2.destroyAllWindows()  # Close all windows


########################################################################################################################
#                                                   MAIN PROGRAM
########################################################################################################################


#samples = np.append(samples,hsv,0)
#Array de imagenes

if training:
    readTraining()

knn = loadTraining()

testAlgorithm(knn)
