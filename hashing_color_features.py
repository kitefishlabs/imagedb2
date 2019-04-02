from color_features_pyramid import *
import numpy as np

def div2(x):
    return int(x/2)

def dfloor(x,d):
    return int(x - (x % d))

class ColorFeaturesHash:
    """
    The aim is to create a hash table mapping colors to points/locations in the image. The colors themselves are binned according to a parameter so that we can control rates of collisions.
    """
    def __init__(self, **analysis_params):
        self.initialize(analysis_params)
    
    def initialize(self, analysis_params=None):
        self._check_cf_params(analysis_params)
        self.image_hash = {}
    
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
            'lower_level' : 2,        # multiple of 2
            'bin_size' : 1,
        }
        return analysis_params

    def load_downsampled(self):
        ap = self.analysis_params
        # access raw data for a pyramid level (downsampling by a power of 2)
        feats = ColorFeaturesPyramid(**{'media_file_path':ap['media_file_path']})
        lofp,imgshp,loshp = feats.get_downsampled_data(ap['lower_level'])
        hifp,_,hishp = feats.get_downsampled_data(ap['lower_level'] + 1)
        # set our hash grid size (for calculating addresses)
        self.grid_height = imgshp[0]
        self.grid_width = imgshp[1]
        # low level (next higher level is half the number of slices/grid points)
        # next lower mult. of 2, so odd nums / 2 do not run off the end
        loh = dfloor(loshp[0], 2)
        low = dfloor(loshp[1], 2)
        b = ap['bin_size']
        # iterate over the lower level/ higher resolution grid
        for row in range(loh):
            for col in range(low):
                pt = lofp[row,col,:]
                hipt = hifp[div2(row),div2(col),:]
                grid_pt = (row, col)
                # by subtracting the modulo (a mult. of 2) the range is preserved and consistent across all pyramid levels
                # bit-shift and add, so that we have a tuple (could be expanded so that index covers more levels)
                binned_colors = (((dfloor(pt[2], b) << 16) + (dfloor(pt[1], b) << 8) + pt[0]), ((dfloor(hipt[2], b) << 16) + (dfloor(hipt[1], b) << 8) + dfloor(hipt[0], b)))
                try:
                    leaf = self.image_hash[binned_colors] + [grid_pt]
                except KeyError:
                    leaf = [grid_pt]
                self.image_hash[binned_colors] = leaf
    
    def check_for_colisions(self):
        """
        simple reporting function to see how many points are collected per key
        """
        for k,val in self.image_hash.items():
            if len(val) > 1:
                print(k)
                print(len(val))
