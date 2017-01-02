import cv2
import numpy as np
from ast import literal_eval

cam = cv2.VideoCapture(0)

set_winname = 'Settings'
cv2.namedWindow(set_winname)
in_winname = 'Raw'
med_winname = 'In-process'
out_winname = 'Out'
settingsfile = "proto_settings.txt"
kernel = np.asarray([[0, 255, 255, 255, 0],
                   [255, 255, 255, 255, 255],
                   [255, 255, 255, 255, 255],
                   [255,255, 255, 255, 255],
                   [0,255, 255, 255, 0]], dtype=np.uint8)

cliprange = [188, 54, 0, 39, 255, 255] # Default values
settings = dict(cliprange=cliprange, target=60)

with open(settingsfile, 'r') as f:
    settings = literal_eval(f.read())
    cliprange = settings['cliprange']

def store_settings():
    with open(settingsfile, 'w') as f:
        print(settings)
        f.write(repr(settings))

def update_cliprange(_):
    sliders = ['Hue_min', 'Sat_min', 'Val_min', 'Hue_max', 'Sat_max', 'Val_max']
    for val, key in enumerate(sliders):
        cliprange[val] = cv2.getTrackbarPos(key, set_winname)
    settings['cliprange'] = cliprange

def update_target(val):
    settings['target'] = val

a = cv2.createTrackbar('Hue_min', set_winname, cliprange[0], 360, update_cliprange)
cv2.createTrackbar('Hue_max', set_winname, cliprange[3], 360, update_cliprange)
cv2.createTrackbar('Sat_min', set_winname, cliprange[1], 255, update_cliprange)
cv2.createTrackbar('Sat_max', set_winname, cliprange[4], 255, update_cliprange)
cv2.createTrackbar('Val_min', set_winname, cliprange[2], 255, update_cliprange)
cv2.createTrackbar('Val_max', set_winname, cliprange[5], 255, update_cliprange)
cv2.createTrackbar('Target', set_winname, settings['target'], 360, update_target)

def cyclicInRange(src, lowerb, upperb):
    if lowerb[0] <= upperb[0]:
        return cv2.inRange(src, lowerb, upperb)
    else:
        lowermask = lowerb.copy()
        uppermask = upperb.copy()
        lowermask[0]=0
        uppermask[0] = upperb[0]
        mask1 = cv2.inRange(src, lowermask, uppermask)
        lowermask[0] = lowerb[0]
        uppermask[0] = 360
        mask2 = cv2.inRange(src, lowermask, uppermask)
        cv2.add(mask1, mask2, mask1)
        return mask1


while True:
    ret, raw = cam.read()
    if not ret:
        print("Failed to capture image.")
        break
    im = cv2.flip(raw, 1)
    cv2.GaussianBlur(im, (7, 7), 5, im)
    cv2.imshow(in_winname, im)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    #im[:, :, 2] = cv2.equalizeHist(im[:,:,2])
    mask = cyclicInRange(hsv, np.array(cliprange[0:3]), np.array(cliprange[3:6]))
    #cv2.erode(mask, kernel, mask, iterations=1)
    #mask = cv2.medianBlur(mask, 7)
    #mask = cv2.medianBlur(mask, 7)
    #mask = cv2.medianBlur(mask, 7)
    #cv2.dilate(mask, kernel, mask, iterations=3)
    cv2.imshow(med_winname, mask)

    if cv2.waitKey(1) != -1:
        print("Interrupted.")
        break
store_settings()
cam.release()
cv2.destroyAllWindows()
