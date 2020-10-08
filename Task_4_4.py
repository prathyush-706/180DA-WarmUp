####--------- IMPROVEMENTS ---------####
# 1. Noticed that my dominant colour was exactly opposite to the color within the rectangle and so converted 
#    color scheme back to BGR from RGB and it worked as expected.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Activate webcam for video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.open()

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

# Coordinates for upper left and bottom right of rectangle
upper_left = (420, 240)
bottom_right = (850, 450)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Draw rectangle around desired region of video stream according to coordiates
    frame = cv2.rectangle(frame,upper_left,bottom_right,(0,0,255),2)
    rect_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    img = cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB)

    # Use Kalman filtering and n_clusters = 1 because we only want the most dominant color
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=1) #cluster number

    # Apply labels
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    # Convert back to BGR color scheme
    bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

    # Display video stream and dominant color
    cv2.imshow('frame',frame)
    cv2.imshow('Dominant color',bar)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()