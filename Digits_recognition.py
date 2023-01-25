import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import imutils
from keras.models import load_model

model = load_model('mnist.h5')

# Testing - with new image
image = cv2.imread('Digits5.png')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
preprocessed_digits = []

for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # Cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y + h, x:x + w]

    # Resizing that digit to (18, 18)
    resized_digit = cv2.resize(digit, (18, 18))

    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

    # Adding the preprocessed digit to the list of preprocessed digits
    preprocessed_digits.append(padded_digit)

# print("Contured Image")
# plt.imshow(image, cmap="gray")
# # plt.xlabel('xlabel')
# # plt.ylabel('ylabel')
# plt.title("Contured digits")
# plt.show()

image_contured = imutils.resize(image, width=480, height=480)
cv2.imshow("Contured digits", image_contured)
cv2.waitKey(0)

inp = np.array(preprocessed_digits)

counter = 1
for digit in preprocessed_digits:
    prediction = model.predict(digit.reshape(1, 28, 28, 1))
    # print(f"\n=========PREDICTION============")
    # plt.imshow(digit.reshape(28, 28), cmap="gray")
    # plt.show()
    digit = digit.reshape(28, 28)
    digit_image = imutils.resize(digit, width=400, height=300)
    cv2.putText(digit_image, "Prediction: {}".format(np.argmax(prediction)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow(f"Digit {str(counter)}" , digit_image)
    cv2.waitKey(0)
    print(f"Digit: {str(counter)}, Prediction: {format(np.argmax(prediction))}")
    counter += 1
    # print("Final Output: {}".format(np.argmax(prediction)))
    # print("Prediction (Softmax) from the neural network:\n {}".format(prediction))
    # hard_maxed_prediction = np.zeros(prediction.shape)
    # hard_maxed_prediction[0][np.argmax(prediction)] = 1
    # print("Hard-maxed form of the prediction: \n {}".format(hard_maxed_prediction))


