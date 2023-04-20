import cv2 as cv
import numpy as np


path = r'C:\Users\jorda\Documents\personal\masters\CSC515 - Computer Vision\Mod6\Discussion\discussion1.png'
img = cv.imread(path)


_, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

cv.imshow("Original Image", img)
cv.imshow("Threshold Image", th1)

cv.waitKey(0)
cv.destroyAllWindows()

