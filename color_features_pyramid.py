# color_features_pyramid.py - color features pyramid data from images
# ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

"""
ColorFeaturesPyramid
Part of ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

Overview
========
todo

"""

import sys, os, math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class ColorFeaturesPyramid:
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
                self.analysis_params[k] = os.path.abspath(self.analysis_params.get(k, dcfp[k]))
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
            'verbose' : False,			# useful for debugging
            'media_file_path' : os.path.abspath('img/gradient.png'),	# default file
            'color_space' : 'lab', 		# 'lab' or 'rgb'
            'down_sampling' : 2,        # multiple of 2
            'onset' : 0,				# time onset in seconds (for time-based media, == 0 for images)
        }
        return analysis_params
    
    def _down_sample_and_analyze(self):
        """
        """
        ap = self.analysis_params
        imgpath = ap['media_file_path']
        print('Down-sample + analyze: ' + imgpath + ' (' + str(ap['down_sampling']) + ')')
        datapath = ap['media_file_path'] + '.pyr'

        img = cv.imread(imgpath)
        imgw = img.shape[1]
        imgh = img.shape[0]
        ds_levels = int(math.log2(ap['down_sampling']))

        imgs_ds = [img] + [np.zeros((int(imgh / math.pow(2, (level+1))), int(imgw / math.pow(2, (level+1))), 3), dtype=np.uint8) for level in range(ds_levels)]

        print(ds_levels)
        print(len(imgs_ds))

        for level in range(ds_levels):
            print('level: ', str(level))
            print(imgs_ds[level].shape)
            print('dst path: ', (datapath + str(level+1)))
            print(imgs_ds[level+1].shape)
            imgs_ds[level+1] = cv.pyrDown(imgs_ds[level], imgs_ds[level+1])
        
        for level in range(ds_levels):
            fp = np.memmap((datapath + str(level+1)), dtype='uint8', mode='w+', shape=(imgs_ds[level+1].shape[0], imgs_ds[level+1].shape[1], 3))
            fp[:] = np.copy(imgs_ds[level+1])
            fp.flush()
        
        return (imgh, imgw)
    
    def load_and_display(self, ds=8):
        ap = self.analysis_params
        imgpath = ap['media_file_path']
        lvl = int(math.log2(ds))
        datapath = ap['media_file_path'] + '.pyr' + str(lvl)
        print('Display down-sampled: ' + datapath)
        img = cv.imread(imgpath)
        fp = np.memmap(datapath, dtype='uint8', mode='r+', shape=(int(img.shape[0] / math.pow(2, lvl)), int(img.shape[1] / math.pow(2, lvl)), 3))
        plt.imshow(fp)
        plt.show()
        return (fp, img.shape, fp.shape)
    
    def get_downsampled_data(self, lvl=8):
        ap = self.analysis_params
        imgpath = ap['media_file_path']
        datapath = ap['media_file_path'] + '.pyr' + str(lvl)
        print('Read down-sampled: ' + datapath)
        img = cv.imread(imgpath)
        fp = np.memmap(datapath, dtype='uint8', mode='r+', shape=(int(img.shape[0] / math.pow(2, lvl)), int(img.shape[1] / math.pow(2, lvl)), 3))
        return (fp, img.shape, fp.shape)
