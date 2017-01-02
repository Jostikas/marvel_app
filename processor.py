"""
*******************************************************************************
    Image processing subroutines for the Marvel App
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
import cv2
import numpy as np

# Define the used datasets. Key should be the same as the corresponding flag in
# settings, value should be a uint8 grayscale image with Cr values on the y axis
# and Cb values on the x axis.
datasets = {
    'dataset_sfa': cv2.imread("makehist/sfa_hist.bmp", cv2.IMREAD_GRAYSCALE),
    'dataset_sfagt': cv2.imread("makehist/sfaGT_hist.bmp", cv2.IMREAD_GRAYSCALE),
    'dataset_ibtd': cv2.imread("makehist/ibtd_hist.bmp", cv2.IMREAD_GRAYSCALE),
    'dataset_uci': cv2.imread("makehist/UCI_hist.bmp", cv2.IMREAD_GRAYSCALE),
}

class Processor(dict):
    """Image processing subroutines for the Marvel App"""

    def __init__(self, settings):
        super(Processor, self).__init__()
        self.settings = settings
        self.compose_hist()

    def process(self, frame, debug=False):
        """Process the frame

        Will perform histogram backprojection, followed by colorization
        of skin-like pixels.

        :param frame: Frame to process
        :param debug: If this is True, will construct and store a debug
        histogram image. It will also preserve the original frame.
        Setting this to False will perform the processing in-place.
        :return: tuple(processed, histogram). If debug is False, then
        histogram = None.
        """
        if debug:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
            # thresh, hist = cv2.threshold(self.hist, self.settings['data_thresh'], 255, cv2.THRESH_BINARY)
            hist = (self.hist / 255).astype(np.float32)
            # hist[128, 128] = 0
            bprj = cv2.calcBackProject([hsv], (0, 1), hist, (0, 256, 0, 256), 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            # cv2.filter2D(bprj, cv2.CV_8U, kernel, bprj)
            threshed = bprj
            cv2.imshow('blurred', threshed)
            threshold, mask = cv2.threshold(threshed, self.settings['skin_thresh'], 255, cv2.THRESH_BINARY)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, mask)
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask)
            # N, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            # for label in range(1,N):
            #     if stats[label, cv2.CC_STAT_AREA] < 300:
            #         mask[labels == label] = 0
            
            cv2.imshow('mask', mask)
            hsv = frame.copy()
            cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV_FULL, hsv)
            s = mask.copy()
            s[s == 0] = hsv[s == 0, 1]
            cv2.GaussianBlur(s, (9, 9), 5, s)
            cv2.imshow('Sat', s)
            hsv[mask > 0, 0] = self.settings['target']
            hsv[:, :, 1] = s

            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL, hsv)
            if self.settings['raw']:
                out = frame
            else:
                out = hsv
            debug_out = hist
        else:
            debug_out = None
            out = frame
        return out, debug_out

    def compose_hist(self):
        """Compose a histogram to use for backprojection.

        The resulting histogram will have the maximum value by pixel of any enabled
         input datasets.

        :return: The composed histogram image
        """
        hist = np.zeros((256,256), np.uint8)
        for key in self.settings.keys():
            if key.startswith('dataset_') and self.settings[key] == True:
                hist = np.max(np.array([hist, datasets[key]]), axis=0)
        self.hist = hist