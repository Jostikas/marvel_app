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
        self.bgimage = np.zeros((480, 640, 3), np.uint8)

    def process(self, frame, debug=False):
        """Process the frame

        Will perform histogram backprojection, followed by colorization
        of skin-like pixels.

        :param frame: Frame to process
        :param debug: If this is True, will display output debug images and return a histogram.
        :return: tuple(processed, histogram). If debug is False, then histogram = None.
        """
        # Do white balance correction
        frame_corr = frame * self.wbgains  # wbgains is found in function set_wb
        cv2.convertScaleAbs(frame_corr, frame)  # Convert float to 8-bit
        # Find skin-colored pixels
        mask = self.find_skin(frame, self.hist, debug)
        #Convert to HSV for colorization
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)  # Convert to HSV
        # get the HS values under the mouse for debugging
        if debug:
            HS_under_mouse = tuple(hsv[self.mouse_loc[::-1]][0:2])
        # Colorize pixels that are determined to be skin
        hsv[mask > 0, 0] = self.settings['target']
        hsv[mask > 0, 1] = 255
        # Convert back to BGR
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

        # Post-processing
        if self.settings['bgimage']:  # If background replacement is active
            out[mask == 0] = self.bgimage[mask == 0]  # Replace the background
        if self.settings['edges']:  # If edge detection is active
            edged = cv2.Laplacian(out, cv2.CV_8U, ksize=5)  # Find edges
            # The kernel used is:
            # 2   4   4   4   2
            # 4   0  -8   0   4
            # 4  -8 -24  -8   4
            # 4   0 - 8   0   4
            # 2   4   4   4   2
            out[mask == 0] = edged[mask == 0]  # Replace background
        if self.settings['raw']:  # If raw is active  (Only in debug mode)
            out = frame  # output raw frame (color corrected, though
        if debug:  # If debug mode is active
            debug_out = self.hist.copy()  # Output a copy of the histogram
            self.draw_gray_marker(debug_out, HS_under_mouse[::-1])  # Draw marker on the histogram
        else:
            debug_out = None
        return out, debug_out

    def find_skin(self, frame, hist, debug=False):
        """Locate skin using backprojection

        If debug is True, will also display the backprojection and output mask.

        :param frame: Input image
        :param hist: Histogram (normalized float32) to backproject
        :return: skin mask
        """
        blurred = cv2.GaussianBlur(frame, (9, 9), 5)  # suppress noise in the image
        self.blurred = blurred
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV_FULL)  # Convert to HSV
        bprj = cv2.calcBackProject([hsv], (0, 1), hist, (0, 256, 0, 256), 255)
        # Threshold the backprojection
        threshold, mask = cv2.threshold(bprj, self.settings['skin_thresh'], 255, cv2.THRESH_BINARY)
        if debug:
            threshed = mask.copy()
        # Use only blobs that are larger than some threshold in the mask
        # Do blob detection
        mask = self._blobsize_threshold(mask, self.settings['blob_thresh'])
        negmask = np.bitwise_not(mask)
        negmask = self._blobsize_threshold(negmask, self.settings['hole_thresh'])
        # cv2.imshow('negmask', negmask)
        mask = np.bitwise_not(negmask)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5), mask)



        if debug:
            cv2.imshow('Backprojection', bprj)
            cv2.imshow('Thresholded and morphed', threshed)
            cv2.imshow('Mask', mask)
        return mask

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
        self.hist = (hist / 255).astype(np.float32) # Convert histogram from uint8 to floating point

    def set_wb(self, swatch):
        graylevel = np.mean(swatch, dtype=np.single)
        means = np.mean(swatch, (0,1), dtype=np.single)
        self.wbgains = graylevel / means

    def set_mouse_loc(self, x, y):
        self.mouse_loc = (x, y)

    def change_custom_hist(self, x, y, add=True, debug=False):
        mask2 = np.zeros((self.blurred.shape[0] + 2, self.blurred.shape[1] + 2), np.uint8)
        cv2.floodFill(self.blurred,
                      mask2,
                      (x, y),
                      255,
                      (10, 10, 10),
                      (10, 10, 10),
                      8 + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY
                      )
        mask = mask2[1:-1, 1:-1]
        hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV_FULL)
        hist = cv2.calcHist([hsv], [0,1], mask,
                            [256, 256], [0,256, 0,256],
                            datasets['dataset_custom'], True)
        cv2.blur(hist, (3, 3), hist)
        hist = cv2.convertScaleAbs(hist, alpha=255/np.max(hist))
        if add:
            cv2.add(datasets['dataset_custom'], hist, datasets['dataset_custom'])
        else:
            cv2.subtract(datasets['dataset_custom'], hist, datasets['dataset_custom'])
        self.compose_hist()
        if debug:
            cv2.imshow('addedmask', hist)

    def reset_custom_hist(self):
        datasets['dataset_custom'] = np.zeros((256, 256), dtype=np.uint8)
        self.compose_hist()

    def draw_gray_marker(self, img, loc):
        """Draws a crosshairs on a grayscale float image

        :param img: Grayscale floating-point [0...1] image
        :param loc: Location to draw the crosshairs at
        """
        graylevel = img[loc[1], loc[0]]
        color = 1 if graylevel < 0.5 else 0
        cv2.circle(img, loc, 3, color)
        cv2.line(img, (loc[0]-5, loc[1]), (loc[0]+5, loc[1]), color)
        cv2.line(img, (loc[0], loc[1]-5), (loc[0], loc[1]+5), color)

    def set_bgimage(self, fname):
        """Set the background image file.

        :param fname: File name of the image
        """
        bgimg = cv2.imread(fname)
        self.bgimage = cv2.resize(bgimg, (640, 480), interpolation=cv2.INTER_CUBIC)

    def _blobsize_threshold(self, img, threshold):
        N, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        # Get the blob numbers that have sizes higher than threshold
        label_list = np.arange(0, N, dtype=np.int32)
        selector = label_list[stats[:, cv2.CC_STAT_AREA] > threshold]
        selector = selector[selector != 0]
        # Form a mask from these large blobs
        mask = np.in1d(labels, selector).astype(np.uint8).reshape(img.shape)
        return mask*255

