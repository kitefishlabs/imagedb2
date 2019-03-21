# color_features.py - color features histogram data from images
# ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

"""
ColorFeatures
Part of ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

Overview
========

Use the color features (currently only L*a*b* and RGB) extractor class to analyze images or frames from video files. The color features class extracts histograms for two frame types. The first is a histogram of color features for the entire image. The second is a set of sixteen histograms, each describing a region of the image. The regions are arranged in an even four-by-four non-overlapping grid, with the first region at the upper left and the last at the lower right. These values are stored in a binary file using Numpy memory-mapped arrays.

Creation and Parameters
=======================

Instantiate the ColorFeatures class, optionally with additional keyword arguments:

.. code-block:: python
    cfl = ColorFeatures (fileName, param1=value1, param2=value2, ...)

The default features-extractor parameters are defined in an internal parameter dictionary. The full list of settable parameters, with default values and explanations:

+------------------------+-----------------+----------------------------------------------------+
| keyword                | default         | explanation                                        |
+========================+=================+====================================================+
| verbose                | True            | useful for debugging                               |
+------------------------+-----------------+----------------------------------------------------+
| media_file_path        | ./img/          | relative to project dir or abs path                |
+------------------------+-----------------+----------------------------------------------------+
| grid_divs              | 4               | number of divisions along each axis                |
+-----------------------------------------------------------------------------------------------+
| onset                  | 0               | time onset in seconds (for time-based media, == 0  |
|                        |                 | for images)                                        |
+------------------------+-----------------+----------------------------------------------------+
| threshold              | 0               | (empirical) threshold for histogram values; set to |
|                        |                 | a positive number to remove extremely low values   |
+------------------------+-----------------+----------------------------------------------------+
| Parameters for color features histograms...                                                   |
+------------------------+-----------------+----------------------------------------------------+
| color_space             | lab             | used for extention naming (.lab + .lab.json)      |
+------------------------+-----------------+----------------------------------------------------+
| dims                   | 16              | number of bins for histogram                       |
+------------------------+-----------------+----------------------------------------------------+
| range                  | 256             | range of data output (starts at 0)                 |
+------------------------+-----------------+----------------------------------------------------+

Parameter keywords can be passed explicitly as formal arguments or as a keyword argument parameter dict:, e.g.:

.. code-block:: python
   cfl = ColorFeaturesLab(filename, range=256, verbose=True )
   cfl = ColorFeaturesLabfilename, **{'range':256, 'verbose':True} )

"""

__version__ = '0.1'
__author__ = 'Thomas Stoll'
__copyright__ = "Copyright (C) 2019 Thomas Stoll, All Rights Reserved"
__license__ = "gpl 3.0 or higher"
__email__ = 'tom@kitefishlabs.com'

import sys, time, os, json
import cv2 as cv
import numpy as np


class ColorFeatures:
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
            'grid_divs' : 4,
            'bins' : 16,				# number of bins for histogram
            'onset' : 0,				# time onset in seconds (for time-based media, == 0 for images)
            'threshold' : 0,			# (empirical) threshold for histogram; set to a positive number to remove extremely low values
        }
        return analysis_params
    
    def _analyze_image(self):
        """
        Analyze full image and gridded image according to params set in class creation.
        Converts BGR color space to Lab color space.
        """
        ap = self.analysis_params
        imgpath = ap['media_file_path']
        print('Analyze ' + imgpath)
        datapath = ap['media_file_path'] + '.' + ap['color_space']
        num_histograms = (ap['grid_divs'] ** 2) + 1

        img = cv.imread(imgpath)
        fp = np.memmap(datapath, dtype='float32', mode='w+', shape=(num_histograms, 3, ap['bins']))

        grid_divs = ap['grid_divs']
        hist_bins = ap['bins']
        hist_range = 256
        thresh = ap['threshold']
        frame_height = int(img.shape[0])
        frame_width = int(img.shape[1])
        grid_width = int(frame_width / grid_divs)
        grid_height = int(frame_height / grid_divs)
        
        # np arrays to hold pixel data after color converson, ititialized with 0s
        img_ = np.zeros((frame_height, frame_width, 3), dtype=np.float32)
        sub_ = np.zeros((grid_height, grid_width, 3), dtype=np.float32)

        # convert color space once
        if ap['color_space'] == 'rgb':
            print('2 rgb')
            img_ = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            print('2 lab')
            img_ = cv.cvtColor(img, cv.COLOR_BGR2Lab)

        # arrays to hold each color dimension's bins
        l_star = np.zeros((hist_bins,1), dtype=np.float32)
        a_star = np.zeros((hist_bins,1), dtype=np.float32)
        b_star = np.zeros((hist_bins,1), dtype=np.float32)

        # analyze entire image - take L2 norm (max = 1.0) each color channel's histogram as we go
        item = cv.calcHist([img_],[0],None,[hist_bins],[0,hist_range])
        cv.normalize(np.where(item>thresh,item,0),l_star)
        item = cv.calcHist([img_],[1],None,[hist_bins],[0,hist_range])
        cv.normalize(np.where(item>thresh,item,0),a_star)
        item = cv.calcHist([img_],[2],None,[hist_bins],[0,hist_range])
        cv.normalize(np.where(item>thresh,item,0),b_star)

        # zeroth slot is full image
        fp[0][0] = np.reshape(l_star[:], (hist_bins))
        fp[0][1] = np.reshape(a_star[:], (hist_bins))
        fp[0][2] = np.reshape(b_star[:], (hist_bins))

        # slice and calculate histograms for sub-images
        for i in range(grid_divs):
            for j in range(grid_divs):
                
                sub_lab = np.zeros((grid_height, grid_width, 3), dtype=np.float32)

                # arrays to hold each color dimension's bins
                l_star = np.zeros((hist_bins,1), dtype=np.float32)
                a_star = np.zeros((hist_bins,1), dtype=np.float32)
                b_star = np.zeros((hist_bins,1), dtype=np.float32)
                
                sub_ = img_[(i*grid_height):((i+1)*grid_height),(j*grid_width):((j+1)*grid_width)]
                                
                item = cv.calcHist([sub_],[0],None,[hist_bins],[0,hist_range])
                cv.normalize(np.where(item>thresh,item,0),l_star)
                item = cv.calcHist([sub_],[1],None,[hist_bins],[0,hist_range])
                cv.normalize(np.where(item>thresh,item,0),a_star)
                item = cv.calcHist([sub_],[2],None,[hist_bins],[0,hist_range])
                cv.normalize(np.where(item>thresh,item,0),b_star)
                        
                fp[ ((grid_divs*i)+j) + 1 ][0] = np.reshape(l_star[:], (hist_bins))
                fp[ ((grid_divs*i)+j) + 1 ][1] = np.reshape(a_star[:], (hist_bins))
                fp[ ((grid_divs*i)+j) + 1 ][2] = np.reshape(b_star[:], (hist_bins))
                
        fp.flush()
        return img.shape


    def get_analysis_result(self):
        """
        print an image's entire data array
        """
        ap = self.analysis_params
        datapath = os.path.abspath(self.analysis_params['media_file_path']+'.'+ap['color_space'])
        print('Access ' + datapath)
        fp = np.memmap(datapath, dtype='float32', mode='r+', shape=(((ap['grid_divs'] ** 2) + 1), 3, ap['bins']))
        return fp