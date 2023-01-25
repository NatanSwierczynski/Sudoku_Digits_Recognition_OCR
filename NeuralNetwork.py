import tensorflow as tf
from keras import models
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

score = model.evaluate(test_images, test_labels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## Saving model to file
# model.save('mnist.h5')
# print("Model saved as mnist.h5")

## Testing - on test data
# example = test_images[random.randint(0, len(test_images))]
# prediction = model.predict(example.reshape(1, 28, 28, 1))
# # First output
# print ("Prediction (Softmax) from the neural network:\n\n {}".format(prediction))
# # Second output
# hard_maxed_prediction = np.zeros(prediction.shape)
# hard_maxed_prediction[0][np.argmax(prediction)] = 1
# print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
# # Third output
# print ("\n\n--------- Prediction --------- \n\n")
# plt.imshow(example.reshape(28, 28), cmap="gray")
# plt.show()
# print("\n\nFinal Output: {}".format(np.argmax(prediction)))

## Testing - with new images
# image = cv2.imread('sudo.png')
# grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
# contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# preprocessed_digits = []
# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)
#
#     # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
#     cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
#
#     # Cropping out the digit from the image corresponding to the current contours in the for loop
#     digit = thresh[y:y + h, x:x + w]
#
#     # Resizing that digit to (18, 18)
#     resized_digit = cv2.resize(digit, (18, 18))
#
#     # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
#     padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
#
#     # Adding the preprocessed digit to the list of preprocessed digits
#     preprocessed_digits.append(padded_digit)
# print("\n\n\n----------------Contoured Image--------------------")
# plt.imshow(image, cmap="gray")
# plt.show()
#
# inp = np.array(preprocessed_digits)
#
# for digit in preprocessed_digits:
#     prediction = model.predict(digit.reshape(1, 28, 28, 1))
#
#     print("\n\n---------------------------------------\n\n")
#     print("=========PREDICTION============ \n\n")
#     plt.imshow(digit.reshape(28, 28), cmap="gray")
#     plt.show()
#     print("\n\nFinal Output: {}".format(np.argmax(prediction)))
#
#     print("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
#
#     hard_maxed_prediction = np.zeros(prediction.shape)
#     hard_maxed_prediction[0][np.argmax(prediction)] = 1
#     print("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
#     print("\n\n---------------------------------------\n\n")
