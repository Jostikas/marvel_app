import cv2
import scipy.misc as misc
import matplotlib.pyplot as plt

works = misc.imread('sfaGT_hist.bmp')
nw = cv2.imread('sfaGT_hist.bmp')

plt.subplot(1,2,1)
plt.imshow(works, interpolation='nearest')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(nw, interpolation='nearest')
plt.colorbar()

plt.show()
