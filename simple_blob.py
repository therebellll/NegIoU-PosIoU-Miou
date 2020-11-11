#!/usr/bin/python

# Standard imports
import cv2
import numpy as np

# Read image
im = cv2.imread("blob.jpg")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

params.minDistBetweenBlobs = 0.0

params.filterByColor = True
params.blobColor = 255

# Change thresholds
params.thresholdStep = 20
params.minThreshold = 100
params.maxThreshold = 140

# Filter by Area.
params.filterByArea = True
params.minArea = 1
params.maxArea = 1800

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')


if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

start = cv2.getTickCount()
# Detect blobs.
keypoints = detector.detect(im)
end = cv2.getTickCount()
during1 = (end - start) / cv2.getTickFrequency()
print(during1)
print(len(keypoints))
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 255, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
