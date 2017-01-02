import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir, chdir, getcwd

"""Using 35x35 px swatches from the SFA dataset.

Stitch all into a single image, then convert it to YCrCb and make 2D histogram over CrCb plane.
Normalize the histogram (with peak value 1).
"""
X = 35  # Swatch measurements
Y = 35

# Read the files list
cwd = getcwd()
chdir('C:/Users/jostikas/Downloads/SFA/SKIN/35')
fileslist = listdir()
N = len(fileslist)

# Allocate the target image
comp = np.empty((Y, N*X, 3), np.uint8)

# Concatenate the images
print('Concatenating...')
for n, path in enumerate(fileslist):
    if not n%(N//10):
        print('*', end='', flush=True)
    comp[:,n*X:n*X+X,:] = cv2.imread(path)
print('...done.')

# Convert image
cv2.cvtColor(comp, cv2.COLOR_BGR2YCrCb, comp)

# Take 2d histogram from the Cr and Cb values.
hist = cv2.calcHist([comp], [1, 2], None, [256, 256], [0,256, 0,256])
hist /= np.max(hist)

# Construct colorplane:
# Create a (256,256,3) image array, where channels 1 and 2 contain the row and column indices.
color_plane = np.flipud(np.indices((256, 256, 1), np.uint8)).T[0]

# Display histogram
gray = cv2.convertScaleAbs(hist, alpha=255)
plt.imshow(gray, interpolation='nearest')
plt.xlim(70,140)
plt.ylim(120, 190)
plt.xlabel('Cb')
plt.ylabel('Cr')
plt.title('SFA swatches')
plt.show()

chdir(cwd)
cv2.imwrite('sfa_hist.bmp', gray)