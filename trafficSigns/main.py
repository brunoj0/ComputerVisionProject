import numpy as np

import cv2
from sklearn.model_selection import train_test_split
import os

path = "myData"
labelFile = 'labels.csv'
testRatio = 0.25
validationRatio = 0.25

#Import Images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
print(path)
noOfClasses=len(myList)
print(len(myList))
print("Importing Classes.....")
for x in range (0,len(myList)):
    # if count > 9:
    #     zeros = '000'
    # else:
    #     zeros = '0000'
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
print(" ")
images = np.array(images)
print(images[0])
classNo = np.array(classNo)

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = image_to_feature_vector(img)
    return img
X_train=np.array(list(map(preprocessing,X_train)))
X_test=np.array(list(map(preprocessing,X_test)))

print("Data Shapes")
print("Train",end = "");print(X_train.shape,y_train.shape)
print("Test",end = "");print(X_test.shape,y_test.shape)
assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"

print("Train",end = "");print(X_train.shape,y_train.shape)


