# color_features_pyramid.py - color features pyramid data from images
# ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

"""
ColorFeaturesPyramid
Part of ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

Overview
========
Pyramid down-sampling + color-feature analysis.
Saves downsampled data to NAME.COLORSPACE.PYR{DOWNSAMPLING} data files mem-mapped to np arrays.
Also retrieves and displays down-sampled image data.

"""

import sys
import os
# import os.path as path
# import re
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import color_features_tiles as cft

class ColorFeaturesPyramid:
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
        if self.params['color_space'] != 'rgb':
            self.params['color_space'] = 'lab'

    @staticmethod
    def default_cf_params():
        params = {
            'media_file_path': os.path.abspath('img/001.jpg'),
            'color_space': 'lab', 		# 'lab' or 'rgb'
            'down_sampling': 2,        # number of pyramid levels 
            'analysis_tileh': 24,
            'analysis_tilew': 36
        }
        return params

    def _determine_num_bins(self, sz, lvl):
            # create a list of np arrays to hold our gradually shrinking arrays
            return int(sz / math.pow(2, (lvl+1)))            

    def _zeros_for_shape_level(self, sz, lvl):
        imgw,imgh = sz
        return np.zeros(
            (self._determine_num_bins(imgh,lvl),
            self._determine_num_bins(imgw,lvl), 
            3), 
            dtype=np.uint8)

    def _up_sample_in_place(self, in_mtrx, pwr, out_shape):    # pwr:  0 1 2 3 4 ...
            print(in_mtrx.shape)
            (a,b,_) = in_mtrx.shape
            out_mtrx = np.zeros(out_shape, dtype=in_mtrx.dtype)
            mult = 2 ** pwr                         # 1 2 4 8 16 ...
            
            # same shape

            # for i in range(a):
            #     for j in range(b):
            #         print(i, j, a, b, in_mtrx[i,j,:])
            
            for i in range(a):
                for j in range(b):
                    # print((i*mult),((i+1)*mult),(j*mult),((j+1)*mult), i, j, a, b, in_mtrx[i,j,:])
                    out_mtrx[(i*mult):((i+1)*mult),(j*mult):((j+1)*mult),:] = in_mtrx[i,j,:]
            return out_mtrx
    
    def _down_sample_and_analyze(self):
        """
        Reads and image, performs a series of down-samplings, writes each to disk
        """
        ap = self.params
        imgpath = ap['media_file_path']
        print('Down-sample + analyze: ' + imgpath +
              ' (' + str(ap['down_sampling']) + ')')
        datapath = ap['media_file_path'] + '.' + ap['color_space'] + '.pyr'

        img = cv.imread(imgpath)
        
        imgw = img.shape[1]
        imgh = img.shape[0]
        ds_levels = int(ap['down_sampling'])

        # convert color space once
        if ap['color_space'] == 'rgb':
            img_ = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            img_ = cv.cvtColor(img, cv.COLOR_BGR2Lab)

        
        
        # add 1 to # of levels so we can write to the Nth + 1 level, properly sized
        # 0th level is image itself in imgs_ds, concat shifts zeros arrays to start
        # at index 1
        imgs_ds = [img_] + [self._zeros_for_shape_level((imgw,imgh), level) for level in range(ds_levels)]

        base_shape = imgs_ds[0].shape
        for level in range(ds_levels):
            imgs_ds[level+1] = cv.pyrDown(imgs_ds[level], imgs_ds[level+1])
            cv.imwrite((datapath + "_" + str(level+1) + ".jpg"), imgs_ds[level+1])
            img_b = self._up_sample_in_place(imgs_ds[level+1], (level+1), base_shape)
            cv.imwrite((datapath + "_" + str(level+1) + ".jpg"), img_b)

        # TODO: split out and refactor to allow different data outputs
        # for level in range(ds_levels):
        #     fp = np.memmap((datapath + str(level+1)), dtype='uint8', mode='w+',
        #                    shape=(imgs_ds[level+1].shape[0], imgs_ds[level+1].shape[1], 3))
        #     fp[:] = np.copy(imgs_ds[level+1])
        #     fp.flush()
        
        # for now we need the 0th, original img AND
        # all the ds_levels recursively down-sampled image files
        # + 1 to include the 0th level
        for level in range(ds_levels+1):
            sub_img = imgs_ds[level]
            feat_tiles = cft.ColorFeaturesTiles(**{
                'media_file_path': (datapath + "_" + str(level+1) + ".jpg"),
                'tileh': 24, 'tilew': 36
            })
            yield (str(level),
                feat_tiles._sample_and_analyze(0),
                feat_tiles._sample_and_analyze(1),
                feat_tiles._sample_and_analyze(2))


    def load_and_display(self, ds=8):
        """
        """
        ap = self.params
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
        ap = self.params
        imgpath = ap['media_file_path']
        datapath = ap['media_file_path'] + '.' + \
            ap['color_space'] + '.pyr' + str(lvl)
        img = cv.imread(imgpath)
        fp = np.memmap(datapath, dtype='uint8', mode='r+', shape=(
            int(img.shape[0] / math.pow(2, lvl)), int(img.shape[1] / math.pow(2, lvl)), 3))
        return fp, img.shape, fp.shape
