import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir, chdir, getcwd

"""Using GT-s from the SFA dataset.

Normalize the histogram (with peak value 1).
"""
BGT = 5  # Background Y threshhold (due to packing artifacts)
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
    cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb, im)
    mask = (im[:,:,0] > BGT).astype(np.uint8)
    temp = cv2.calcHist([im], [1, 2], mask, (256, 256), (0, 256, 0, 256))
    cv2.accumulate(temp, hist)
hist /= np.max(hist)
print('...Done.')

# Display histogram
gray = cv2.convertScaleAbs(hist, alpha=255)
plt.imshow(gray, interpolation='nearest')
plt.xlim(70,140)
plt.ylim(120, 190)
plt.xlabel('Cb')
plt.ylabel('Cr')
plt.title('SFA GTs')
plt.show()

chdir(cwd)
cv2.imwrite('sfaGT_hist.bmp', gray)