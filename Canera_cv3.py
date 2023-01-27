import cv2
import numpy as np
from keras.models import load_model
import imutils
from imutils.perspective import four_point_transform
from Sudoku_opencv import find_puzzle, extract_digit, sudoku_ocr

# initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # capture an image from the camera
    ret, frame = cap.read()

    ## convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # find contours in the thresholded image and sort them by size in descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we can assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break

    # if the puzzle contour is empty then our script could not find the outline of the Sudoku puzzle so raise an error
    # if puzzleCnt is None:
    #     raise Exception(("Could not find Sudoku puzzle outline"))

    # draw the contour of the puzzle on the image and then display it to our screen for visualization/debugging purposes
    output = frame.copy()
    cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
    cv2.imshow("6.Puzzle Outline", output)

    # check to see if we are visualizing the perspective transform
    # puzzle = four_point_transform(frame, puzzleCnt.reshape(4, 2))
    # warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    # cv2.imshow("7.Puzzle Transform", puzzle)
    # cv2.imshow("8.Puzzle Transform gray", warped)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite("camera_sudoku.jpg", frame)
        print("Image saved!")
    elif key == ord('q'):
        break

# release the camera and close the window
cap.release()
cv2.destroyAllWindows()

model = load_model('mnist.h5')
image_name = 'camera_sudoku.jpg'
image = cv2.imread(image_name)
show_image_proccessing = True
show_digits_extract = True
image_final = sudoku_ocr(model, image, show_image_proccessing, show_digits_extract)
cv2.imshow("10.Sudoku Result", image_final)
cv2.waitKey(0)




