import cv2
import numpy as np


def _blobsize_threshold(img, threshold):
    N, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    # Get the blob numbers that have sizes higher than threshold
    label_list = np.arange(0, N, dtype=np.int32)
    selector = label_list[stats[:, cv2.CC_STAT_AREA] > threshold]
    selector = selector[selector != 0]
    # Form a mask from these large blobs
    mask = np.in1d(labels, selector).astype(np.uint8).reshape(img.shape)
    return mask * 255

hue = 95
sat = 255
img = cv2.imread('Pictures/Processed.png')
exp = cv2.imread('Pictures/andromeda-006.jpg')

exp = cv2.resize(exp, (640, 480), interpolation=cv2.INTER_LANCZOS4)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
mask = (hsv[:,:,:2] == [hue, sat])[:,:,0].astype(np.uint8)*255
mask = _blobsize_threshold(mask, 100)
#cv2.medianBlur(mask, 5, mask)
invmask = cv2.bitwise_not(mask)
invmask = _blobsize_threshold(invmask, 100)
mask = cv2.bitwise_not(invmask)
lap = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
out = img.copy()
out[mask==0] = lap[mask==0]
cv2.imshow('blah', out)
cv2.waitKey(0)
cv2.imwrite('Edged.png', out)
out = img.copy()
out[mask==0] = exp[mask==0]
cv2.imshow('blah', out)
cv2.waitKey(0)
cv2.imwrite('BG.png', out)


