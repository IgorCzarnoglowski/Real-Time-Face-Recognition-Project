import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=5)


videoCap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

dataset_path = './face_dataset/'

face_data = []
labels = []
class_id = 0
names = {} #mapping between id and name

'dataset preparation'

for fx in os.listdir(dataset_path): #we are looping through dataset directory
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        target = class_id + np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)


