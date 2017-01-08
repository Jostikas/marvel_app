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
    'dataset_custom': np.zeros((256, 256), np.uint8)
}

class Processor(dict):
    """Image processing subroutines for the Marvel App"""

    def __init__(self, settings):
        super(Processor, self).__init__()
        self.settings = settings
        self.compose_hist()
        self.wbgains = np.array([1,1,1], np.single)
        self.mouse_loc = (0,0)

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
        # Do white balance correction
        frame_corr = frame * self.wbgains  # wbgains is found in function set_wb
        cv2.convertScaleAbs(frame_corr, frame)  # Convert float to 8-bit
        self.frame = frame
        blurred = cv2.GaussianBlur(frame, (9,9), 5)  # Reduce noise in the image
        if debug:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV_FULL) # Convert to HSV
            # thresh, hist = cv2.threshold(self.hist, self.settings['data_thresh'], 255, cv2.THRESH_BINARY)
            hist = (self.hist / 255).astype(np.float32) # Convert histogram from uint8 to floating point
            # hist[128, 128] = 0
            bprj = cv2.calcBackProject([hsv], (0, 1), hist, (0, 256, 0, 256), 255)
            cv2.imshow('blurred', bprj)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            threshold, mask = cv2.threshold(bprj, self.settings['skin_thresh'], 255, cv2.THRESH_BINARY)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, mask)
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask)
            # N, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            # for label in range(1,N):
            #     if stats[label, cv2.CC_STAT_AREA] < 300:
            #         mask[labels == label] = 0
            
            cv2.imshow('mask', mask)
            hsv = frame.copy()
            cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV_FULL, hsv)
            HS_under_mouse = tuple(hsv[self.mouse_loc[::-1]][0:2])
            # s = mask.copy()
            # s[s == 0] = hsv[s == 0, 1]
            # cv2.GaussianBlur(s, (9, 9), 5, s)
            # cv2.imshow('Sat', s)
            hsv[mask > 0, 0] = self.settings['target']
            hsv[mask > 0, 1] = 255
            # hsv[:, :, 1] = s
            # hsv[:, :, 1] = 255

            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL, hsv)
            if self.settings['edges']:
                edged = cv2.Laplacian(hsv, cv2.CV_8U, ksize=5)
                hsv[mask == 0] = edged[mask == 0]
            if self.settings['raw']:
                out = frame
            else:
                out = hsv
            debug_out = hist
            self.draw_gray_marker(debug_out, HS_under_mouse[::-1])

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

    def set_wb(self, swatch):
        graylevel = np.mean(swatch, dtype=np.single)
        means = np.mean(swatch, (0,1), dtype=np.single)
        self.wbgains = graylevel / means

    def set_mouse_loc(self, x, y):
        self.mouse_loc = (x, y)

    def change_custom_hist(self, x, y, add=True):
        mask2 = np.zeros((self.frame.shape[0] + 2, self.frame.shape[1] + 2), np.uint8)
        cv2.floodFill(self.frame,
                      mask2,
                      (x, y),
                      255,
                      (20, 20, 20),
                      (20, 20, 20),
                      8 + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY
                      )
        mask = mask2[1:-1, 1:-1]
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV_FULL)
        hist = cv2.calcHist([hsv], [0,1], mask,
                     [256, 256], [0,256, 0,256],
                     datasets['dataset_custom'], True)
        cv2.morphologyEx(hist, cv2.MORPH_OPEN, (3,3), hist)
        cv2.blur(hist, (3, 3), hist)
        hist = cv2.convertScaleAbs(hist, alpha=255/np.max(hist))
        cv2.imshow('addedmask', hist)
        if add:
            cv2.add(datasets['dataset_custom'], hist, datasets['dataset_custom'])
        else:
            cv2.subtract(datasets['dataset_custom'], hist, datasets['dataset_custom'])
        self.compose_hist()

    def reset_custom_hist(self):
        datasets['dataset_custom'] = np.zeros((256, 256), dtype=np.uint8)
        self.compose_hist()


    def draw_gray_marker(self, img, loc):
        graylevel = img[loc[1], loc[0]]
        color = 255 if graylevel < 127 else 0
        cv2.circle(img, loc, 3, color)
        cv2.line(img, (loc[0]-5, loc[1]), (loc[0]+5, loc[1]), color)
        cv2.line(img, (loc[0], loc[1]-5), (loc[0], loc[1]+5), color)

