import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load skin data values. The values are from 'Skin Segmentation Dataset', UCI Machine Learning Repository, by
# Rajen Bhatt and Abhinav Dhall. Only using the values marked as "skin".
skindata = np.loadtxt('skindata.txt', np.uint8, usecols=(0,1,2))

# Add a dimension (shape from [50859, 3] to [1, 50859, 3], so it makes an image matrix as far as openCV is concerned.
skindata = skindata[None,:,:]
cv2.cvtColor(skindata, cv2.COLOR_BGR2HSV_FULL, skindata)  # Is actually YCbCr ???

## Create the lookup table, with Cb as the x axis and Cr as the y axis.
lookup = np.zeros((256,256))  # dtype default is float64

for pixel in skindata[0,:,0:2]:
    lookup[pixel[0], pixel[1]] += 1

cv2.normalize(lookup, lookup, alpha=np.sum(lookup)/np.max(lookup)/2)

lookup_im = cv2.convertScaleAbs(lookup, alpha=255)
cv2.blur(lookup_im, (7,7), lookup_im)

plt.imshow(lookup_im, interpolation='nearest')
plt.xlabel('S')
plt.ylabel('H')
plt.title('UCI pixels')
plt.show()

cv2.imwrite('UCI_hist.bmp', lookup_im)
