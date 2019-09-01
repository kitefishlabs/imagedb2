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


class ColorFeaturesTiles:
    """
    """

    def __init__(self, **analysis_params):
        self.initialize(analysis_params)

    def initialize(self, analysis_params=None):
        self._check_cf_params(analysis_params)

    def _check_cf_params(self, analysis_params=None):
        """
        Simple mechanism to read in default parameters while substituting custom parameters.
        """
        self.analysis_params = analysis_params if analysis_params is not None else self.analysis_params
        dcfp = self.default_cf_params()
        for k in dcfp.keys():
            if k == 'media_file_path':
                self.analysis_params[k] = os.path.abspath(
                    self.analysis_params.get(k, dcfp[k]))
            elif (k == 'color_space'):
                # enforce lab-unless-rgb for color_space
                if self.analysis_params.get(k, 'lab') != 'rgb':
                    self.analysis_params[k] = 'lab'
                else:
                    self.analysis_params[k] = 'rgb'
            else:
                self.analysis_params[k] = self.analysis_params.get(k, dcfp[k])
        return self.analysis_params

    @staticmethod
    def default_cf_params():
        analysis_params = {
            'verbose': False,			# useful for debugging
            # default file
            'media_file_path': os.path.abspath('img/gradient.png'),
            'color_space': 'lab', 		# 'lab' or 'rgb'
            'down_sampling': 2,        # multiple of 2
            # time onset in seconds (for time-based media, == 0 for images)
            'onset': 0,
        }
        return analysis_params

    def _sample_and_analyze(self):
        """
        Reads and image, performs a series of samplings, feature extraction, writes each to disk
        """
        ap = self.analysis_params
        imgpath = ap['media_file_path']
        print('Sample + analyze: ' + imgpath +
              ' (' + str(ap['down_sampling']) + ')')
        # datapath = ap['media_file_path'] + '.' + ap['color_space'] + '.pyr'

        m = re.search(r'(\d+).jpg', path.basename(imgpath))
        idnum = m.group(1)

        img = cv.imread(imgpath)
        imgw = img.shape[1]
        imgh = img.shape[0]
        tilesize = ap['tile-size']
        # tileolap = ap['tile-overlap']

        # convert color space once
        if ap['color_space'] == 'rgb':
            img = cv.cvtColor(img, cv.COLOR_RGB2RGB)
        else:
            img = cv.cvtColor(img, cv.COLOR_RGB2Lab)

        aspect = imgw / imgh
        tilew = int(img.shape[0] / tilesize)
        tileh = int(img.shape[1] / tilesize)

        # print("Tile sizes:")
        # print(tilew)
        # print(tileh)
        # print(aspect)

        mask = np.zeros(img.shape[:2], np.uint8)

        all_histograms = np.array([], dtype=np.float32)
        all_histograms.shape = (0, 1)

        for y in range(tilesize):
            for x in range(tilesize):

                # print("tilesize, x, y:")
                # print(tilesize)
                # print(y)
                # print(x)
                # print("coords:")
                # print((tileh * y))
                # print((tileh * (y + 1)))
                # print((tilew * x))
                # print((tilew * (x + 1)))

                mask[(tilew * x):(tilew * (x + 1)),
                     (tileh * y):(tileh * (y + 1))] = 255
                masked_img = cv.bitwise_and(img, img, mask=mask)

                hist_masked = cv.calcHist([img], [0], mask, [32], [0, 256])
                hist_masked = cv.normalize(hist_masked, hist_masked)
                hist_masked.shape = (32, 1)

                id_info = np.array(
                    [idnum, x, y, tilew, tileh], dtype='float32')
                id_info.shape = (5, 1)

                # print(hist_masked.shape)
                # print(id_info.shape)

                # print(hist_masked.dtype)
                # print(id_info.dtype)

                row = np.r_[hist_masked, id_info]
                # print(row.shape)
                all_histograms = np.r_[all_histograms, row]

                # plt.subplot(221), plt.imshow(img, 'gray')
                # plt.subplot(222), plt.imshow(mask, 'gray')
                # plt.subplot(223), plt.imshow(masked_img, 'gray')
                # plt.subplot(224), plt.plot(hist_full), plt.plot(hist_masked)
                # plt.xlim([0, 32])
                # plt.show()

        # print("::")
        # print(all_histograms.shape)
        all_histograms.shape = (37, int(all_histograms.shape[0] / 37))
        # print(all_histograms.shape)

        return all_histograms

    def load_and_display(self, ds=8):
        """
        """
        ap = self.analysis_params
        imgpath = ap['media_file_path']
        lvl = int(math.log2(ds))
        datapath = ap['media_file_path'] + '.' + \
            ap['color_space'] + '.pyr' + str(lvl)
        img = cv.imread(imgpath)
        fp = np.memmap(datapath, dtype='uint8', mode='r+', shape=(
            int(img.shape[0] / math.pow(2, lvl)), int(img.shape[1] / math.pow(2, lvl)), 3))
        plt.imshow(fp)
        plt.show()

    def get_downsampled_data(self, lvl=8):
        """
        """
        ap = self.analysis_params
        imgpath = ap['media_file_path']
        datapath = ap['media_file_path'] + '.' + \
            ap['color_space'] + '.pyr' + str(lvl)
        img = cv.imread(imgpath)
        fp = np.memmap(datapath, dtype='uint8', mode='r+', shape=(
            int(img.shape[0] / math.pow(2, lvl)), int(img.shape[1] / math.pow(2, lvl)), 3))
        return fp, img.shape, fp.shape
