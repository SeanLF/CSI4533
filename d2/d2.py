import numpy as np
import cv2


# Call me with the file name or 0 for webcam
def run(fileName):
    cv2.namedWindow("Display frame", cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(fileName)
    while(True if fileName is 0 else cap.isOpened()):
        _, frame = cap.read()
        __processFrame(frame)
        cv2.imshow('Display frame', frame)
        cv2.waitKey(1)
    # cleanup
    cap.release()
    cv2.destroyAllWindows()


def __processFrame(frame):
    # blur and convert to grayscale to perform hough circle transform
    img = cv2.medianBlur(frame, 11)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect circles in blurred grayscale frame
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 70, param1=50, param2=30, minRadius=0, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, i in enumerate(circles[0, :]):
            # draw the outer circle with thick green lines
            if idx < 5:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255 - idx*50, 0), 2)
            else:
                break
