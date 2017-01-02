import cv2
import numpy as np

gimg = np.empty((480,640,3), np.uint8)

def flood(event, x, y, n, m):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Event: Button down ({}, {})".format(x, y))
        mask2 = np.zeros((gimg.shape[0] + 2, gimg.shape[1] + 2), np.uint8)
        cv2.floodFill(gimg, mask2, (x, y), 255, (20, 20, 20), (20,20,20), 8+cv2.FLOODFILL_FIXED_RANGE)
        cv2.imshow('mask', mask2)


cam = cv2.VideoCapture(0)
win = cv2.namedWindow('window', True)
cv2.setMouseCallback('window', flood)


while True:
    ret, img = cam.read()
    gimg[...] = img[...]
    cv2.waitKey(0)
    cv2.imshow('window', gimg)
