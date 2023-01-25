import cv2
import numpy as np
from keras.models import load_model

# load the trained model
model = load_model('mnist.h5')

# initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # capture an image from the camera
    ret, frame = cap.read()

    # preprocess the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = np.reshape(gray, (1, 28, 28, 1))

    # apply thresholding to the image
    threshold, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255-img_bin

    # find contours in the image
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # loop through the contours and draw them on the image
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # predict the digit using the model
    digit = model.predict(gray)
    digit = np.argmax(digit)

    # display the predicted digit on the screen
    cv2.putText(frame, str(digit), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close the window
cap.release()
cv2.destroyAllWindows()
