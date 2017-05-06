import cv2
import numpy as np
import matplotlib.pyplot as plt

trainData = np.random.randint(0,100,(51,2)).astype(np.float32)
responses = np.random.randint(0,2,(51,1)).astype(np.float32)

train = np.empty((0,6), dtype=np.float32)
train_labels = np.empty((0,1), dtype=np.float32)

np.append(train_labels, np.array([[float("1")]]), axis=0)
np.append(train_labels, np.array([[float("2")]]), axis=0)
np.append(train_labels, np.array([[float("3")]]), axis=0)

train = np.append(train, np.array([[1, 2, 3, 1, 2, 3]]), axis=0)
train = np.append(train, np.array([[1, 2, 4, 1, 2, 3]]), axis=0)
train = np.append(train, np.array([[1, 2, 53, 1, 2, 3]]), axis=0)

'''
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')'''


newcomer = np.random.randint(0,100,(5,2)).astype(np.float32)
#plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

print ("ret: ", ret,"\n")
print ("results: ", results,"\n")
print ("neighbours: ", neighbours,"\n")
print ("distances: ", dist)

newc = np.empty((0,6), dtype=np.float32)
newc = np.append(newc, np.array([[1, 2, 53, 1, 2, 3]]), axis=0)

knn2 = cv2.ml.KNearest_create()
knn2.train(np.float32(train),cv2.ml.ROW_SAMPLE,np.float32(train_labels))
ret, results, neighbours, dist = knn2.findNearest(np.float32(newc), 1)

print ("ret: ", ret,"\n")
print ("results: ", results,"\n")
print ("neighbours: ", neighbours,"\n")
print ("distances: ", dist)
#plt.show()