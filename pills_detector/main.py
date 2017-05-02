import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def definer(hsv, imagen, l, h):
    # HSV bounds para el color rojo
    low = np.array(l)
    high = np.array(h)

    # Crea una imagen binaria basada en HSV
    mask = cv2.inRange(hsv, low, high)
    # Elimina los puntos pequenos
    mask = cv2.medianBlur(mask, 7)
    # Combina la imagen binaria con el color de la imagen original
    new_image = cv2.bitwise_and(imagen, imagen, mask=mask)
    # Coge los contornos de la mascara roja
    img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return new_image


# main functionality to train the program with the images
# Empty Array for the labels
samples =  np.empty((0,300))
#Array vacio para los valores correspondientes
valueDigit = []

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

def loadSamples():
    # Feature set containing (x,y) values of 300 known/training data
    trainData = np.random.randint(0, 100, (300, 2)).astype(np.float32)

    # Labels each one either Red or Blue with numbers 0 and 1
    responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

    # Take Red families and plot them
    red = trainData[responses.ravel() == 0]
    plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

    # Take Blue families and plot them
    blue = trainData[responses.ravel() == 1]
    plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

    plt.show()

    #trainData = np.loadtxt('generalsamples.data',np.float32)
	#responses = np.loadtxt('generalvaluedigits.data',np.float32)
	#responses = responses.reshape((responses.size,1))

	# cv2.KNearest.train(trainData, responses[, sampleIdx[, isRegression[, maxK[, updateBase]]]]) → retval
    # Parameters:
    # isRegression – Type of the problem: true for regression and false for classification.
    # maxK – Number of maximum neighbors that may be passed to the method CvKNearest::find_nearest().
    # updateBase – Specifies whether the model is trained from scratch (update_base=false), or it is updated using the new training data (update_base=true). In the latter case, the parameter maxK must not be larger than the original value.
    # Utilizamos el algoritmo KNeartest
    model = cv2.KNearest()
    # Entrenamos el algoritmo
    model.train(trainData, responses)
    return model

for key in data:
    print (key, data[key])

    class_name = key
    directory = data[key][0]
    first_color = data[key][1]
    second_color = None
    if len(data[key]) > 2:
        second_color = data[key][2]

    print ("Class " + class_name + " -> " + directory)

    files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.lower().endswith(".jpg")]

    for f in files:
        print("Image: " + f)
        imagen = cv2.imread(directory + f)
        blur = cv2.GaussianBlur(imagen, (3, 3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        cv2.imshow("Color Image, class " + class_name, imagen)
        cv2.moveWindow("Color Image, class " + class_name, 0, 0)

        low =  np.array(first_color[0])
        high = np.array(first_color[1])

        color_mask = definer(hsv, imagen, low, high)

        cv2.imshow("Mask 1, class " + class_name, color_mask)
        cv2.moveWindow("Mask 1, class " + class_name, 1000, 0)

        if second_color != None:
            low2 = np.array(second_color[0])
            high2 = np.array(second_color[1])
            color_mask2 = definer(hsv, imagen, low2, high2)
            cv2.imshow("Mask 2, class " + class_name, color_mask2)
            cv2.moveWindow("Mask 2, class " + class_name, 1000, 1000)

        th3 = cv2.adaptiveThreshold(cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        cv2.imshow("adaptiveThreshold" + class_name, th3)

        cv2.waitKey(5000)  # Wait for a key
        # Asigna el valor de la imagen
        valueDigit.append(key)

    cv2.destroyAllWindows()  # Close all windows


#samples = np.append(samples,hsv,0)
#Array de imagenes
valueDigit = np.array(valueDigit,np.float32)
valueDigit = valueDigit.reshape((valueDigit.size,1))
print ("Fin de la fase de entrenamiento")

# Fase de test
print("Fase de test")

kNearestNeighbourhood = loadSamples()
#testAlgorithm(kNearestNeighbourhood)
