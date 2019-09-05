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
            'tile_divs': 2,        # multiple of 2
            # time onset in seconds (for time-based media, == 0 for images)
            'onset': 0,
        }
        return analysis_params

    def _sample_and_analyze(self, channel=0):
        """
        Reads and image, performs a series of samplings, feature extraction, writes each to disk
        """
        ap = self.analysis_params
        imgpath = ap['media_file_path']
        print('Sample + analyze: ' + imgpath +
              ' (' + str(ap['tile_divs']) + ')')
        # datapath = ap['media_file_path'] + '.' + ap['color_space'] + '.pyr'

        idnum = -1
        m = re.search(r'(\d+).[jpg|png]', path.basename(imgpath))
        if m is not None:
            idnum = m.group(1)

        img = cv.imread(imgpath)
        imgw = img.shape[1]
        imgh = img.shape[0]
        tilesize = ap['tile_divs']
        # tileolap = ap['tile-overlap']

        tilesh = int(math.floor(imgh / 36))
        tilesw = int(math.floor(imgh / 72))

        # convert color space once
        if ap['color_space'] == 'rgb':
            img = cv.cvtColor(img, cv.COLOR_RGB2RGB)
        else:
            img = cv.cvtColor(img, cv.COLOR_RGB2Lab)

        aspect = imgw / imgh
        tilew = 72  # int(img.shape[0] / tilesize)
        tileh = 36  # int(img.shape[1] / tilesize)

        print("Tile sizes:")
        print(tilew)
        print(tileh)
        print(aspect)

        mask = np.zeros(img.shape[:2], np.uint8)

        all_histograms = np.array([], dtype=np.float32)
        all_histograms.shape = (0, 1)

        for y in range(tilesh):
            for x in range(tilesw):

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

                hist_masked = cv.calcHist([img], [channel], mask, [
                                          16], [0, 256])
                hist_masked = cv.normalize(hist_masked, hist_masked)
                # print(hist_masked.shape)
                hist_masked.shape = (16, 1)

                id_info = np.array(
                    [idnum, x, y, tilew, tileh], dtype='float32')
                id_info.shape = (5, 1)

                # print(hist_masked.shape)
                # print(id_info.shape)

                # print(hist_masked)
                # print(id_info)

                row = np.r_[hist_masked, id_info]
                # print(row.shape)
                all_histograms = np.r_[all_histograms, row]

                # plt.subplot(221), plt.imshow(img, 'gray')
                # plt.subplot(222), plt.imshow(mask, 'gray')
                # plt.subplot(223), plt.imshow(masked_img, 'gray')
                # plt.subplot(224), plt.plot(hist_masked)  # plt.plot(hist_full),
                # plt.xlim([0, 8])
                # plt.show()

        # print("::")
        # print(all_histograms.shape)
        all_histograms.shape = (
            int(all_histograms.shape[0] / (16 + 5)), (16 + 5))

        return all_histograms

    # def load_and_display(self, ds=8):
    #     """
    #     """
    #     ap = self.analysis_params
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
    #     ap = self.analysis_params
    #     imgpath = ap['media_file_path']
    #     datapath = ap['media_file_path'] + '.' + \
    #         ap['color_space'] + '.pyr' + str(lvl)
    #     img = cv.imread(imgpath)
    #     fp = np.memmap(datapath, dtype='uint8', mode='r+', shape=(
    #         int(img.shape[0] / math.pow(2, lvl)), int(img.shape[1] / math.pow(2, lvl)), 3))
    #     return fp, img.shape, fp.shape
