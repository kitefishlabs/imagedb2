# color_features_tiles.py - color features tiled data from images
# ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

"""
ColorFeaturesTiles
Part of ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

Overview
========
Pyramid down-sampling + color-feature analysis.
Saves downsampled data to NAME.COLORSPACE.TILE{SIZE}.OLAP{SIZE} data files mem-mapped to np arrays.
Also retrieves and displays tiled image data.

"""

import sys
import os
import os.path as path
import re
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

NUM_HISTO_BINS = 16
NUM_ID_BINS = 5 # id, x, y, tilew, tileh

class ColorFeaturesTiles:
    """
    """

    def __init__(self, **params):
        self.initialize(params)

    def initialize(self, params=None):
        self._check_cf_params(params)
        self._post_check_cf_params()

    def _check_cf_params(self, params=None):
        """
        Simple mechanism to read in default parameters while substituting custom parameters.
        """
        self.params = params if params is not None else {}
        dcfp = self.default_cf_params()
        for k in dcfp.keys():
            self.params[k] = self.params.get(k, dcfp[k])
        return self.params

    def _post_check_cf_params(self):
        """
        Checking values for correctness/files for existence/etc and changing to defaults if incorrect.
        """
        self.params['imgid'] = self._extract_integer_from_img_filename(self.params['media_file_path'])
        if self.params['color_space'] != 'rgb':
            self.params['color_space'] = 'lab'

    @staticmethod
    def default_cf_params():
        params = {
            'verbose': False,			                       # useful for debugging
            'media_file_path': os.path.abspath('img/000.jpg'), # let's say this is the default
            'tileh': 24,
            'tilew': 36,
            'color_space': 'lab', 		                       # 'lab' or 'rgb'
        }                                                      # 'lab' is default!
        return params

    def _extract_integer_from_img_filename(self, imgpath: str):
        """
        we are assuming the image fits the pattern NNNN.jpg or .png where there can be any amount of zero padding
        """
        img_idnum = -1
        m = re.search(r'(\d+).[jpg|png]', path.basename(imgpath))
        if m is not None:
            img_idnum = m.group(1)
        return img_idnum

    def _sample_and_analyze(self, channel=0):
        """
        Reads and image, performs a series of samplings, feature extraction, writes each to disk
        """
        ap = self.params
        tilew,tileh = ap['tilew'],ap['tileh']
        imgpath = ap['media_file_path']
        print('Sample + analyze: ' + imgpath +
              ' (' + str(tilew) + ', ' + str(tileh) + ')')
        
        img = cv.imread(imgpath)
        
        imgw = img.shape[1]
        imgh = img.shape[0]
        tilesh = int(math.floor(imgh / tileh))
        tilesw = int(math.floor(imgw / tilew))
        aspect = imgw / imgh

        # convert color space once
        if ap['color_space'] == 'rgb':
            img = cv.cvtColor(img, cv.COLOR_RGB2RGB)
        # elif ap['color_space'] == 'gray':
        #     img = cv.cvtColor(img, cv.COLOR_RGB2Grayscale)
        else:
            img = cv.cvtColor(img, cv.COLOR_RGB2Lab)
        
        mask = np.zeros(img.shape[:2], np.uint8)

        all_histograms = np.array([], dtype=np.float32)
        all_histograms.shape = (0, 1)

        # iterate over rows and cols of grid (dependent on 
        # tile size) 
        lastx, lasty = 0,0
        for y in range(tilesh):
            for x in range(tilesw):
 
                # this is skipped only on the very first iteration,
                # runs on every remaining tile/iteration
                # if x and y have advanced and as they continue to advance,
                # lastx and lasty will trail and zero out mask rectangles
                if ((lastx,lasty) != (x,y)):
                    mask[(tilew * lastx):(tilew * (lastx + 1)),
                     (tileh * lasty):(tileh * (lasty + 1))] = 0
                lastx = x
                lasty = y
                # open the mask rect area only
                mask[(tilew * x):(tilew * (x + 1)),
                     (tileh * y):(tileh * (y + 1))] = 255
                
                hist_masked = cv.calcHist(
                    [img], 
                    [channel], 
                    mask, 
                    [NUM_HISTO_BINS], 
                    [0, 256])
                hist_masked.shape = (NUM_HISTO_BINS, 1)
                hist_masked_normed = hist_masked[:]
                
                hist_masked = cv.normalize(
                    hist_masked, 
                    hist_masked,
                    norm_type=cv.NORM_L2)
                
                # This is the index "appendage" to the hisogram
                # each has a unique id assigned
                # Together (16 + 5 values) they constitute a row
                # a row in the output is a single color channel's histogram
                # over a tile within the image (16 bins, 256 range, bin size 16)
                id_info = np.array(
                    [ap['imgid'], x, y, tilew, tileh], dtype='float32')
                id_info.shape = (NUM_ID_BINS, 1)
                row = np.r_[hist_masked, id_info]
                all_histograms = np.r_[all_histograms, row]

        all_histograms.shape = (int(all_histograms.shape[0] / (NUM_HISTO_BINS + NUM_ID_BINS)), (NUM_HISTO_BINS + NUM_ID_BINS))
    
        return all_histograms

    # def load_and_display(self, ds=8):
    #     """
    #     """
    #     ap = self.params
    #     imgpath = ap['media_file_path']
    #     lvl = int(math.log2(ds))
    #     datapath = ap['media_file_path'] + '.' + \
    #         ap['color_space'] + '.pyr' + str(lvl)
    #     img = cv.imread(imgpath)
    #     fp = np.memmap(datapath, dtype='uint8', mode='r+', shape=(
    #         int(img.shape[0] / math.pow(2, lvl)), int(img.shape[1] / math.pow(2, lvl)), 3))
    #     plt.imshow(fp)
    #     plt.show()

    # def get_downsampled_data(self, lvl=8):
    #     """
    #     """
    #     ap = self.params
    #     imgpath = ap['media_file_path']
    #     datapath = ap['media_file_path'] + '.' + \
    #         ap['color_space'] + '.pyr' + str(lvl)
    #     img = cv.imread(imgpath)
    #     fp = np.memmap(datapath, dtype='uint8', mode='r+', shape=(
    #         int(img.shape[0] / math.pow(2, lvl)), int(img.shape[1] / math.pow(2, lvl)), 3))
    #     return fp, img.shape, fp.shape
