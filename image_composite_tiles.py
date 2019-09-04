# image_composite_tiles.py - color features tiled data from images
# ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

"""
ImageCompositeTiles
Part of ImageDB2 - Tools for analysis, indexing, search, and creative use of image data

Overview
========
Manages target image and coordinates corpus.
Manages parameters and state for image modifications.
"""

import sys
import os
import os.path as path
import random
import pickle
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt


class ImageCompositeTiles:
    """
    """

    def __init__(self, **params):
        self.initialize(params)
        self.target_image = None
        self.overlay_image = None
        self.manip_array = None
        self.th = self.tw = self.tth = self.ttw = 0
        self.generations = 0
        self.runid = 0

    def initialize(self, params=None):
        self._check_cf_params(params)

    def _check_cf_params(self, params=None):
        """
        Simple mechanism to read in default parameters while substituting custom parameters.
        """
        self.params = params if params is not None else self.params
        dcfp = self.default_cf_params()
        for k in dcfp.keys():
            if k == 'media_file_path':
                self.params[k] = os.path.abspath(
                    self.params.get(k, dcfp[k]))
            elif (k == 'color_space'):
                # enforce lab-unless-rgb for color_space
                if self.params.get(k, 'color_space') != 'rgb':
                    self.params[k] = 'lab'
                else:
                    self.params[k] = 'rgb'
            else:
                self.params[k] = self.params.get(k, dcfp[k])
        return self.params

    @staticmethod
    def default_cf_params():
        params = {
            'verbose': False,			# useful for debugging
            # default dir
            'image_input_dir': os.path.abspath('img/'),
            'image_output_dir': os.path.abspath('out/'),
            'color_space': 'lab', 		# 'lab' or 'rgb'
            'tile_divs': 16,            # multiple of 2
            'tick_time': 2.0,
            'target_id': 1,
            'ops': ['no-op', 'add-random-tile', 'add-simmilar-internal', 'add-similar-external', 'add-neighbor-adjacent'],
            'percentages': [0.5, 1.0, 0.52, 0.55, 1.0],
            'ghost_target': False,
        }
        return params

    def padint(self, val, places=4):
        assert(val >= 0)
        vlen = len(str(val))
        places = max(vlen, places)
        diff = places - vlen
        if diff == 1:
            return '0' + str(val)
        elif diff == 2:
            return '00' + str(val)
        elif diff == 3:
            return '000' + str(val)
        elif diff == 4:
            return '0000' + str(val)
        elif diff == 5:
            return '00000' + str(val)
        else:
            return str(val)

    def choose_op(self):
        r = random.random()
        if r < self.params['percentages'][0]:
            return 0
        elif r < self.params['percentages'][1]:
            return 1
        elif r < self.params['percentages'][2]:
            return 2
        elif r < self.params['percentages'][3]:
            return 3
        else:
            return 4

    def init_target(self, tid):
        p = self.params
        if tid > 9:
            path = p['image_input_dir'] + '/0' + str(tid) + '.jpg'
        else:
            path = p['image_input_dir'] + '/00' + str(tid) + '.jpg'
        self.target_image = cv.imread(path)
        target_histo = pickle.loads(open((path + ".histo"), "rb").read())
        self.ttw = int(target_histo[0][7])
        self.tth = int(target_histo[0][6])
        self.tw = self.ttw * 16
        self.th = self.tth * 16

    def init_modifications(self):
        self.manip_array = [[None for r in range(self.params['tile_divs'])]
                            for c in range(self.params['tile_divs'])]

    def init_overlay(self, tid):
        p = self.params
        self.generations = 0
        if tid > 9:
            path = p['image_input_dir'] + '/0' + str(tid) + '.jpg'
        else:
            path = p['image_input_dir'] + '/00' + str(tid) + '.jpg'
        # copy target if ghost == False
        if self.params['ghost_target'] is False:
            self.overlay_image = cv.imread(path)
        else:
            self.overlay_image = np.ones((self.th, self.tw, 3)) * 255.0

    def write_overlay_image(self, outpath):
        assert(self.overlay_image is not None)
        cv.imwrite(outpath, self.overlay_image)

    def modify_rule1(self, gen):
        # ROW COL CHAN
        rando_target_loc = (random.randint(0, 15), random.randint(0, 15))
        rando_sampled_loc = (random.randint(0, 15), random.randint(0, 15))
        rando_sampled_id = 3  # random.randint(0, 9)
        # update modifications array
        self.manip_array[rando_target_loc[0]][rando_target_loc[1]] = (gen,
                                                                      2, rando_sampled_id,
                                                                      rando_sampled_loc[0], rando_sampled_loc[1])
        print([[data for data in row if data is not None]
               for row in self.manip_array])

        # load overlay histo file to grab dimmensions
        simg_histo_path = self.params['image_input_dir'] + \
            '/' + self.padint(rando_sampled_id, 3) + '.jpg.histo'
        simg_histo = pickle.loads(open(simg_histo_path, 'rb').read())
        sy, sx = rando_sampled_loc[0], rando_sampled_loc[1]
        sth, stw = simg_histo[0][6], simg_histo[0][7]
        ty, tx = rando_target_loc

        # print(oimg_histo)
        print(rando_target_loc)
        print(sy)
        print(sx)
        print(self.tth)
        print(self.ttw)
        print(sth)
        print(stw)
        print(ty)
        print(tx)

        # # load overlay image and extract the tile subimage
        simg_path = self.params['image_input_dir'] + \
            '/' + self.padint(rando_sampled_id, 3) + '.jpg'
        simg = cv.imread(simg_path)

        # until the resolutions match, this is going to clip (or underfill at far edges)
        oh1 = int(ty * self.tth)
        oh2 = int(oh1 + self.tth)
        ow1 = int(tx * self.ttw)
        ow2 = int(ow1 + self.ttw)

        print("-------")
        print(oh1)
        print(oh2)
        print(ow1)
        print(ow2)

        oimg_path = self.params['image_output_dir'] + '/' + \
            self.padint(gen, 5) + '.jpg'
        print(oimg_path)

        sh1 = int(sth * sy)
        sh2 = int(sh1 + min(self.tth, sth))
        sw1 = int(stw * sx)
        sw2 = int(sw1 + min(self.ttw, stw))

        print("=========")
        print(sh1)
        print(sh2)
        print(sw1)
        print(sw2)
        oh2 = int(oh1 + min(self.tth, sth))
        ow2 = int(ow1 + min(self.ttw, stw))
        print("<<<<<>>>>>")
        print(oh2)
        print(ow2)
        self.overlay_image[oh1:oh2, ow1:ow2, :] = simg[sh1:sh2, sw1:sw2, :]
        self.write_overlay_image(oimg_path)

    def modify_rule2(self):
        pass

    def modify_rule3(self):
        pass

    def modify_rule4(self):
        pass

    def init_images_and_data(self, runid=0):
        self.runid = runid
        p = self.params
        tid = p['target_id']

        self.init_target(tid)
        self.init_overlay(tid)
        self.init_modifications()

    # def get_target_path(self, path, idnum):
    #     return(path + '/img_' + padint(self.generation, 4) + '.jpg')

    def update_for_n_generations(self, n=10):
        p = self.params
        for gen in range(n):
            # op = 0
            op = self.choose_op()
            if op == 1:
                self.modify_rule1((self.generations + gen))
            else:
                pass
            output_path = p['image_output_dir'] + \
                '/' + self.padint((self.generations + gen), 5) + '.jpg'
            cv.imwrite(output_path, self.overlay_image)
        self.generations += n
        print("Done! Use ffmpeg to create your animation...")
