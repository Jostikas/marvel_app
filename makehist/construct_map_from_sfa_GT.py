import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir, chdir, getcwd

"""Using GT-s from the SFA dataset.

Normalize the histogram (with peak value 1).
"""
BGT = 3  # Background Y threshhold (due to packing artifacts)
X = 3  # Swatch measurements
Y = 3

# Read the files list
cwd = getcwd()
chdir('C:/Users/jostikas/Downloads/SFA/GT')
fileslist = listdir()
N = len(fileslist)

hist = np.zeros((256,256))

# Convert and calculate histogram of each image.
print('Processing...')
for n,path in enumerate(fileslist):
    if n % (N//10) == 0:
        print('*', end='', flush=True)
    im = cv2.imread(path)
    cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL, im)
    mask = (im[:,:,2] > BGT).astype(np.uint8)
    mask[im[:,:,1] == 255] = 0
    temp = cv2.calcHist([im], [0, 1], mask, (256, 256), (0, 256, 0, 256))
    cv2.accumulate(temp, hist)
hist /= np.max(hist)
print('...Done.')

# Display histogram
gray = cv2.convertScaleAbs(hist, alpha=255)
cv2.blur(gray, (5, 5), gray)
plt.imshow(gray, interpolation='nearest')
plt.xlabel('S')
plt.ylabel('H')
plt.title('SFA GTs')
plt.show()

chdir(cwd)
cv2.imwrite('sfaGT_hist.bmp', gray)