import cv2
from keras.models import load_model
from Sudoku_opencv import find_puzzle, extract_digit, sudoku_ocr

import numpy as np
import imutils

from sudoku import Sudoku
from keras.utils import img_to_array

### Odpalanie okienka z obrazem z kamery i zamkniecie go na przycisk ###
# webcam=cv2.VideoCapture(0)
#
# while True:
#     ret,frame=webcam.read()
#
#     if ret==True:
#         cv2.imshow("Camera Window",frame)
#         key=cv2.waitKey(20) & 0xFF
#         if key==ord("q"):
#             break
#
# webcam.release()
# cv2.destroyAllWindows()

model = load_model('mnist.h5')
# print("Model loaded")

show_image_proccessing = False
show_digits_extract = False

cap = cv2.VideoCapture(0)
width, height, fps = 1280/2, 720/2, 15
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)
print(f"Starting the video stream: {width}x{height}")

while True:
    # Capture images frame-by-frame
    ret, frame = cap.read()
    frame_orig = frame.copy()
    cv2.imshow("Camera Window",frame_orig)
    try:
        # Process
        #res, img_out = sudoku_ocr(model, frame)
        image_final = sudoku_ocr(model, frame, show_image_proccessing, show_digits_extract)
        cv2.imshow("Sudoku Result", image_final)
        # Display
        #cv2.imshow('Frame', img_out)
    except Exception as e:
        # If something was wrong, save frame for debugging
        cv2.imwrite("crash.png", frame_orig)
        break

    # image_final = sudoku_ocr(model, frame, show_image_proccessing, show_digits_extract)
    # cv2.imshow("Sudoku Result", image_final)

    # Process key codes
    key_code = cv2.waitKey(1)
    if key_code & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
