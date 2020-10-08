####--------- IMPROVEMENTS ---------####
# 1. Changed the HSV values to better detect blue colors
# 2. Did not convert to grayscale but instead performed adaptive thresholding on the 'mask' image
#    which highlights only the blue parts of the image.
# 3. Used contourArea function to only detect significant patches of blue in the frame and to eliminate some background detection.
# 4. Video capture alone wasn't working for me so I explicitly used cap.open() to activate the camera.



import numpy as np
import cv2

#Activating the camera to enable video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.open()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #BGR Values
    lower_blue = np.array([125, 0, 0])
    upper_blue = np.array([255, 210, 130])

    mask = cv2.inRange(frame, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # Adaptive thresholding on mask to convert to a binary image to make finding contours easier
    thresh = cv2.adaptiveThreshold(mask, 125, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    contours, hierarchy = cv2.findContours(thresh,1,2)

    #Try to eliminate background detection by looking for significant patches of color
    for i in contours:
    	if cv2.contourArea(i) > 200:
    		x, y, w, h = cv2.boundingRect(i)
    		frame = cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
    
    #Display video streams
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()