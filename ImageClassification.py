import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models


(training_images, training_labels), (testing_images,testing_labels) = datasets.cifar10.load_data()

#scale the data from 0-255 to 0-1

training_images, testing_images = training_images/255, testing_images/225


class_names = ['Plane', 'Car' , 'Bird' , 'Cat' , 'Deer' , ' Dog' , 'Frog', 'Horse' , 'Ship' ,'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])          #below each image there is a label

plt.show()


training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]



#build neural network

#model = models.Sequential()
#model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (32,32,3)))
#model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Conv2D(64, (3,3), activation='relu'))
#model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Conv2D(64, (3,3) , activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(10, activation='softmax')) #scales results so they add up to 1


#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#model.fit(training_images, training_labels, epochs = 10, validation_data=(testing_images,testing_labels)) # epochs specifies how many times the model will see the data, here it will see the image 10 times


#loss, accuracy = model.evaluate(testing_images, testing_labels)
#print(f"Loss = {loss}")
#print(f"Accuracy = {accuracy}")

#model.save('image_classifier.model')

model = models.load_model('image_classifier.model')

img = cv.imread('horse.jpg')
img = cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array[img] / 255)
index = np.argmax(prediction)

#argmax gives us the index of the maximum value 

print(f'prediction is {class_names[index]}')





