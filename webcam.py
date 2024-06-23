import cv2
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

y = []
images = []

num_classes = int(input('Number of classes: \n')) 
labels = [];

for i in range(0, num_classes):
   name = input(f'Label name for class N. {i}: \n') 
   labels.append(name)

while True:    
    ret, frame = cap.read()
    cv2.imshow('webcam', frame)

    k = cv2.waitKey(1)

    frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), axis=0)
    frame = frame / 255

    cap_img = False
    
    if k == ord("q"):
        break

    for i in range(0, num_classes):
        if k == ord(str(i)):
            classe = i
            cap_img = True       

    if k == ord('t'):        
        images = np.array(images)
        y = np.array(y)
        y = y.ravel()
        y = to_categorical(y, num_classes)

        images = images.reshape((len(images), np.prod(images.shape[1:])))
        dimInput = images.shape[1]
        
        model = Sequential()

        model.add(Dense(units = 150, input_dim=dimInput, activation='relu'))
        model.add(Dense(units = 150, activation='relu'))
        model.add(Dense(units = num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        

        model.fit(images, y, epochs=100, shuffle=True)

    if k == ord("p"):          
        imagesP = np.array([frame])
        imagesP = imagesP.reshape((len(imagesP), np.prod(imagesP.shape[1:]))) 

        predict = model.predict(imagesP)
        predict_label = list(np.where(predict[0] >= 0.5, 1, 0))

        try:
            print("Identified class: ", labels[int(predict_label.index(1))])
        except:
            print("Unable to recognize class.")    

        print("Probabilities: ")
        for i in range(0, num_classes):
            label_name = labels[i]
            percent = round(predict[0][i]*100)
            print(f"{label_name}: {percent}%")

    if k == ord("h"):
        print(labels)

    if cap_img:
        print("Class selected: ", classe)

        images.append(frame)
        y.append(classe)


cap.release()
cv2.destroyAllWindows()
