"""
*******************************************************************************
    GUI and mainloop logic for the Marvel App
    Copyright (C) 2016  Laur Joost <daremion@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

sfa = cv2.imread('sfa_hist.bmp', cv2.IMREAD_GRAYSCALE)
sfaGT = cv2.imread('sfaGT_hist.bmp', cv2.IMREAD_GRAYSCALE)
UCI = cv2.imread('UCI_hist.bmp', cv2.IMREAD_GRAYSCALE)
ibtd = cv2.imread('ibtd_hist.bmp', cv2.IMREAD_GRAYSCALE)

combined = np.max(np.array((sfa, ibtd), dtype=np.uint8), axis=0)

def plothist(hist, title):
    plt.imshow(hist, interpolation='nearest')
    plt.grid('on')
    plt.xlabel('S')
    plt.ylabel('H')
    plt.title(title)


plt.subplot(221)
plothist(sfaGT, "SFA GTs")
plt.subplot(222)
plothist(ibtd, "ibtd GTs")
plt.colorbar()
plt.subplot(223)
plothist(sfa, "SFA swatches")
plt.subplot(224)
plothist(UCI, "UCI pixels")
plt.colorbar()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.35)
plt.show()

plothist(combined, 'Combined SFA and ibtd')
plt.show()

cv2.imwrite('combined.bmp', combined)

