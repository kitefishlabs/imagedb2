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
from math import floor
import pickle
import cv2 as cv
import numpy as np
from utils import padint
# import matplotlib.pyplot as plt
from image_tiles_hash import *


class ImageCompositeTiles:
    """
    """

    def __init__(self, **params):
        self.initialize(params)
        self.target_id = 0
        self.target_image = None
        self.overlay_image = None
        self.manip_array = None
        self.th = self.tw = self.tth = self.ttw = 0
        self.imgw = self.imgh = 0
        self.numh = self.numw = 0
        self.generations = 0
        self.runid = 0
        self.scan_test_cell = 0
        self.image_tiles_hash = ImageTilesHash() 
        self.cell_choices = []

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
            'percentages': [0.0, 0.5, 1.0, 1.0, 1.0],
            'ghost_target': False,
        }
        return params

    

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
        self.target_id = tid
        p = self.params
        if tid > 9:
            path = p['image_input_dir'] + '/0' + str(tid) + '.jpg'
        else:
            path = p['image_input_dir'] + '/00' + str(tid) + '.jpg'
        self.target_image = cv.imread(path)
        target_histo = pickle.loads(open((path + ".histo"), "rb").read())

        self.ttw = int(target_histo[-1][6])
        self.tth = int(target_histo[-1][7])
        assert(self.ttw == 72)
        assert(self.tth == 36)
        self.imgw = self.target_image.shape[1]
        self.imgh = self.target_image.shape[0]
        self.numw = int(floor(self.imgw / self.ttw))
        self.numh = int(floor(self.imgh / self.tth))
        self.tw = self.numw * 72
        self.th = self.numh * 36
        assert(self.tw <= self.imgw)
        assert(self.th <= self.imgh)

        print("INIT>>>>>>>>>>")
        print(path)
        print((self.imgh, self.imgw))
        print((self.tth, self.ttw))
        print((self.numh, self.numw))
        print((self.th, self.tw))

    def init_modifications(self):
        self.manip_array = [[None for r in range(self.numw)]
                            for c in range(self.numh)]

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
        # print(self.overlay_image)
        if self.overlay_image is not None:
            cv.imwrite(outpath, self.overlay_image)

    def modify_scan_test(self, gen):
        # ROW COL CHAN
        # rando_target_loc = (random.randint(0, self.numh - 1),
                            # random.randint(0, self.numw - 1))
        rando_sampled_loc = (random.randint(0, self.numh - 1),
                             random.randint(0, self.numw - 1))
        rando_sampled_id = 3

        rando_target_loc = (floor(gen / self.numw), (gen % self.numw))

        print(rando_target_loc)
        # print((range(self.numh), range(self.numw)))

        # random.randint(0, 9)
        # update modifications array
        try:
            self.manip_array[rando_target_loc[0]][rando_target_loc[1]] = (gen,
                                                                          2, rando_sampled_id,
                                                                          rando_sampled_loc[0],
                                                                          rando_sampled_loc[1])
        except IndexError:
            print("skipping this generation's modification due to IndexError")
            return

        # print([[data for data in row if data is not None]
            #    for row in self.manip_array])

        # load overlay histo file to grab dimmensions
        simg_histo_path = self.params['image_input_dir'] + \
            '/' + padint(rando_sampled_id, 3) + '.jpg.histo'
        simg_histo = pickle.loads(open(simg_histo_path, 'rb').read())
        sy, sx = rando_sampled_loc[0], rando_sampled_loc[1]
        stw, sth = simg_histo[0][6], simg_histo[0][7]
        ty, tx = rando_target_loc

        # print(rando_target_loc)
        # print(sy)
        # print(sx)
        # print(self.tth)
        # print(self.ttw)
        # print(sth)
        # print(stw)
        # print(ty)
        # print(tx)

        # # load overlay image and extract the tile subimage
        simg_path = self.params['image_input_dir'] + \
            '/' + padint(rando_sampled_id, 3) + '.jpg'
        simg = cv.imread(simg_path)

        # until the resolutions match, this is going to clip (or underfill at far edges)
        oh1 = int(ty * self.tth)
        oh2 = int(oh1 + self.tth)
        ow1 = int(tx * self.ttw)
        ow2 = int(ow1 + self.ttw)

        # print("-------")
        # print(oh1)
        # print(oh2)
        # print(ow1)
        # print(ow2)

        oimg_path = self.params['image_output_dir'] + '/' + \
            padint(gen, 5) + '.jpg'
        # print(oimg_path)

        sh1 = int(sth * sy)
        sh2 = int(sh1 + min(self.tth, sth))
        sw1 = int(stw * sx)
        sw2 = int(sw1 + min(self.ttw, stw))

        # print("=========")
        # print(sh1)
        # print(sh2)
        # print(sw1)
        # print(sw2)
        oh2 = int(oh1 + min(self.tth, sth))
        ow2 = int(ow1 + min(self.ttw, stw))
        # print("<<<<<>>>>>")
        # print(oh2)
        # print(ow2)
        try:
            self.overlay_image[oh1:oh2, ow1:ow2, :] = simg[sh1:sh2, sw1:sw2, :]
            self.write_overlay_image(oimg_path)
        except ValueError:
            print("skipping this generation's image overlay write due to ValueError")

    def modify_rule1(self, gen):
        # Flip a random tile
        # ROW COL CHAN
        rando_target_loc = (random.randint(0, self.numh - 1),
                            random.randint(0, self.numw - 1))
        
        rando_sampled_id = random.randint(0, 26)
        
        # load overlay image to extract the tile subimage
        simg_path = self.params['image_input_dir'] + \
            '/' + padint(rando_sampled_id, 3) + '.jpg'
        simg = cv.imread(simg_path)

        snumh = int(floor(simg.shape[0] / 36))
        snumw = int(floor(simg.shape[1] / 72))

        rando_sampled_loc = (random.randint(0, snumh - 1),
                             random.randint(0, snumw - 1))

        # update modifications array
        try:
            self.manip_array[rando_target_loc[0]][rando_target_loc[1]] = (gen,
                                                                          2, rando_sampled_id,
                                                                          rando_sampled_loc[0],
                                                                          rando_sampled_loc[1])
        except IndexError:
            print("skipping this generation's modification due to IndexError")
            return

        # print([[data for data in row if data is not None]
            #    for row in self.manip_array])

        # load overlay histo file to grab dimmensions
        simg_histo_path = self.params['image_input_dir'] + \
            '/' + padint(rando_sampled_id, 3) + '.jpg.histo'
        simg_histo = pickle.loads(open(simg_histo_path, 'rb').read())
        sy, sx = rando_sampled_loc
        stw, sth = simg_histo[0][6], simg_histo[0][7]
        ty, tx = rando_target_loc

        # load output image to overlay the tile subimage
        oimg_path = self.params['image_output_dir'] + '/' + \
            padint(gen, 5) + '.jpg'
        # print(oimg_path)

        # until the resolutions match, this is going to clip (or underfill at far edges)
        oh1 = int(ty * self.tth)
        oh2 = int(oh1 + self.tth)
        ow1 = int(tx * self.ttw)
        ow2 = int(ow1 + self.ttw)

        sh1 = int(sy * self.tth)
        sh2 = int(sh1 + self.tth)
        sw1 = int(sx * self.ttw)
        sw2 = int(sw1 + self.ttw)

        try:
            self.overlay_image[oh1:oh2, ow1:ow2, :] = simg[sh1:sh2, sw1:sw2, :]
            self.write_overlay_image(oimg_path)
        except ValueError:
            print("skipping this generation's image overlay write due to ValueError")
            print("-------")
            print(oh1)
            print(oh2)
            print(ow1)
            print(ow2)
            print("=========")
            print(sh1)
            print(sh2)
            print(sw1)
            print(sw2)
            print("<<<<<>>>>>")
            print(self.imgh)
            print(self.imgw)
            print(simg.shape)

    def nn2f(self, x):
        return (1. - x) ** 4.
    
    def nnf(self, x):
        return self.nn2f(x)

# from image_composite_tiles import *
# ict = ImageCompositeTiles()
# ict.init_images_and_data()
# ict.update_for_n_generations(200)

    def modify_rule2(self, gen):
        
        # Flip random tile in target to simmilar tile chosen using nn/nn2 indexes
        
        p = self.params
        tid = p['target_id']
        timg_path = p['image_input_dir'] + '/' + \
            str(padint(tid, 3)) + '.jpg'

        timg_nn2 = self.image_tiles_hash.lookup_image_index(tid)
        
        

        ordered_distrib = [self.nn2f(d[0]) for d in timg_nn2]
        data = [i for i in range(len(timg_nn2))]

        print(len(ordered_distrib))
        print(len(data))
        print(ordered_distrib[:20])
        print(data[:20])

        rando_target_id = random.choices(
            data, weights=ordered_distrib, k=1)[0]

        # (dist, (to-id, from-r, from-c))
        timg_nn2_cell = timg_nn2[rando_target_id]
        rando_target_loc = (timg_nn2_cell[1][2], timg_nn2_cell[1][3])

        rando_sampled_id = timg_nn2_cell[1][1]

        cell_choices = self.image_tiles_hash.lookup_image_tile_index(tid, timg_nn2_cell[1])

        print(">>>\n")
        # print(timg_nn2_cell)
        # print("")
        print(rando_target_id)
        print("")
        print(cell_choices)
        print("")
        
        choice_weights = [self.nnf(item[-1]) for item in cell_choices]
        chosen_cell = random.choices(cell_choices, weights=choice_weights, k=1)[0]
        chosen_id = chosen_cell[1]
        assert(chosen_id == rando_sampled_id)
        chosen_r = chosen_cell[4]
        chosen_c = chosen_cell[5]

        print("=============>>>")
        print(chosen_cell)
        print(chosen_id)
        print(chosen_r)
        print(chosen_c)

        # load overlay histo file to grab dimmensions
        simg_path = self.params['image_input_dir'] + \
            '/' + padint(chosen_id, 3) + '.jpg'
        simg_histo_path = simg_path + '.histo'
        simg = cv.imread(simg_path)
        if simg is not None:
            print("Simg:::")
            print(simg.shape)
        try:
            simg_histo = pickle.loads(open(simg_histo_path, 'rb').read())
        except FileNotFoundError:
            print("Skipping bad histo file!")
            print(simg_histo_path)
            return

        sy, sx = (chosen_r, chosen_c)
        stw, sth = (simg_histo[0][6], simg_histo[0][7])
        ty, tx = rando_target_loc

        # update modifications array
        try:
            self.manip_array[int(ty)][int(tx)] = (gen,
                                                  2, rando_sampled_id,
                                                  int(sy),
                                                  int(sx))
        except IndexError:
            print("skipping this generation's modification due to IndexError")
            return

        oimg_path = self.params['image_output_dir'] + '/' + \
            padint(gen, 5) + '.jpg'
        # print(oimg_path)

        # until the resolutions match, this is going to clip (or underfill at far edges)
        oh1 = int(ty * self.tth)
        oh2 = int(oh1 + self.tth)
        ow1 = int(tx * self.ttw)
        ow2 = int(ow1 + self.ttw)

        sh1 = int(sy * self.tth)
        sh2 = int(sh1 + self.tth)
        sw1 = int(sx * self.ttw)
        sw2 = int(sw1 + self.ttw)

        try:
            self.overlay_image[oh1:oh2, ow1:ow2, :] = simg[sh1:sh2, sw1:sw2, :]
            self.write_overlay_image(oimg_path)
        except ValueError:
            print("skipping this generation's image overlay write due to ValueError")
            print("-------")
            print(oh1)
            print(oh2)
            print(ow1)
            print(ow2)
            print("=========")
            print(sh1)
            print(sh2)
            print(sw1)
            print(sw2)
            print("<<<<<>>>>>")
            print(self.imgh)
            print(self.imgw)
            print(simg.shape)

    def modify_rule2a(self, gen):
        
        # Flip random tile in target to simmilar tile chosen using nn/nn2 indexes
        
        p = self.params
        tid = p['target_id']
        timg_path = p['image_input_dir'] + '/' + \
            str(padint(tid, 3)) + '.jpg'

        timg_nn2 = self.image_tiles_hash.lookup_image_index(tid)
        
        

        ordered_distrib = [self.nn2f(d[0]) for d in timg_nn2]
        data = [i for i in range(len(timg_nn2))]

        # print(len(ordered_distrib))
        # print(len(data))
        # print(ordered_distrib[:20])
        # print(data[:20])

        rando_target_id = random.choices(
            data, weights=ordered_distrib, k=1)[0]

        # (dist, (to-id, from-r, from-c))
        timg_nn2_cell = timg_nn2[rando_target_id]
        rando_target_loc = (timg_nn2_cell[1][2], timg_nn2_cell[1][3])

        rando_sampled_id = timg_nn2_cell[1][1]

        self.cell_choices = self.image_tiles_hash.lookup_image_tile_index(tid, timg_nn2_cell[1])

        # print(">>>\n")
        # # print(timg_nn2_cell)
        # # print("")
        # print(rando_target_id)
        # print("")
        # print(self.cell_choices)
        # print("")
        

        


        # choice_weights = [self.nnf(item[-1]) for item in cell_choices]
        
        for i,chosen_cell in enumerate(self.cell_choices):
        
            # chosen_cell = random.choices(cell_choices, weights=choice_weights, k=1)[0]
            chosen_id = chosen_cell[1]
            assert(chosen_id == rando_sampled_id)
            chosen_r = chosen_cell[4]
            chosen_c = chosen_cell[5]

            # print("=============>>>")
            # print(chosen_cell)
            # print(chosen_id)
            # print(chosen_r)
            # print(chosen_c)

            # load overlay histo file to grab dimmensions
            simg_path = self.params['image_input_dir'] + \
                '/' + padint(chosen_id, 3) + '.jpg'
            simg_histo_path = simg_path + '.histo'
            simg = cv.imread(simg_path)
            if simg is not None:
                print("Simg:::")
                print(simg.shape)
            try:
                simg_histo = pickle.loads(open(simg_histo_path, 'rb').read())
            except FileNotFoundError:
                print("Skipping bad histo file!")
                print(simg_histo_path)
                return

            sy, sx = (chosen_r, chosen_c)
            stw, sth = (simg_histo[0][6], simg_histo[0][7])
            ty, tx = rando_target_loc
            vari = i

            # update modifications array
            try:
                self.manip_array[int(ty)][int(tx)] = (gen,vari,
                                                    2, rando_sampled_id,
                                                    int(sy),
                                                    int(sx))
            except IndexError:
                print("skipping this generation's modification due to IndexError")
                return 0

            oimg_path = self.params['image_output_dir'] + '/' + \
                padint(gen, 5) + '_' + padint(i, 3) + '.jpg'
            # print(oimg_path)

            # until the resolutions match, this is going to clip (or underfill at far edges)
            oh1 = int(ty * self.tth)
            oh2 = int(oh1 + self.tth)
            ow1 = int(tx * self.ttw)
            ow2 = int(ow1 + self.ttw)

            sh1 = int(sy * self.tth)
            sh2 = int(sh1 + self.tth)
            sw1 = int(sx * self.ttw)
            sw2 = int(sw1 + self.ttw)

            try:
                self.overlay_image[oh1:oh2, ow1:ow2, :] = simg[sh1:sh2, sw1:sw2, :]
                self.write_overlay_image(oimg_path)
            except ValueError:
                print("skipping this generation's image overlay write due to ValueError")
                print("-------")
                print(oh1)
                print(oh2)
                print(ow1)
                print(ow2)
                print("=========")
                print(sh1)
                print(sh2)
                print(sw1)
                print(sw2)
                print("<<<<<>>>>>")
                print(self.imgh)
                print(self.imgw)
                print(simg.shape)
                return 0
                
        return(len(self.cell_choices))

    def modify_rule3(self, gen):
        # ROW COL CHAN
        # rando_target_loc = (floor(gen / self.numw), (gen % self.numw))
        rando_target_loc = (random.randint(0, self.numh - 1),
                            random.randint(0, self.numw - 1))

        # print(rando_target_loc)
        # print((range(self.numh), range(self.numw)))

        # lookup in modif array
        state = self.manip_array[rando_target_loc[0]][rando_target_loc[1]]

        if state is None:
            latest_id = self.target_id
        else:
            latest_id = state[2]

        # choose direction
        index_delta = random.choice([(1,0), (0,1), (-1,0), (0,-1)])
        # update manip_array
        # caclulate unbounded target location
        calculated_target = (rando_target_loc[0] + index_delta[0], rando_target_loc[1] + index_delta[1])
        
        # check limits - the base image
        if (calculated_target[0] > self.numh) or (calculated_target[1] > self.numw):
            print("Target off edge. Skipping")
            return

        # load potential overlay histo file to grab dimmensions
        limg_path = self.params['image_input_dir'] + \
            '/' + padint(latest_id, 3) + '.jpg'
        limg_histo_path = limg_path + '.histo'
        
        try:
            limg_histo = pickle.loads(open(limg_histo_path, 'rb').read())
        except FileNotFoundError:
            print("Skipping bad file!")
            print(limg_histo_path)
            return

        limg = cv.imread(limg_path)
        # if limg is not None:
        #     print("Limg:::")
        #     print(limg.shape)
        # return limg_histo
        
        limg_h,limg_w = (int(floor(limg.shape[0] / 36)), int(floor(limg.shape[1] / 72)))

        # check limits - the latest/linked image
        # calculated_target = (min(max(calculated_target[0],0), self.numh), min(max(calculated_target[1],0), self.numw))
        if (calculated_target[0] > limg_h) or (calculated_target[1] > limg_w):
            print("Linked image would run off edge. Skipping")
            return

        # update modifications array
        try:
            self.manip_array[rando_target_loc[0]][rando_target_loc[1]] = (gen,
                                                                          3, latest_id,
                                                                          calculated_target[0],
                                                                          calculated_target[1])
        except IndexError:
            print("skipping this generation's modification due to IndexError")
            return

        # update output image
        # stw, sth = simg_histo[0][6], simg_histo[0][7]
        ty, tx = rando_target_loc
        sy, sx = calculated_target
        
        # print(rando_target_loc)
        # print(sy)
        # print(sx)
        # print(self.tth)
        # print(self.ttw)
        # print(sth)
        # print(stw)
        # print(ty)
        # print(tx)

        # until the resolutions match, this is going to clip (or underfill at far edges)
        oh1 = int(ty * self.tth)
        oh2 = int(oh1 + self.tth)
        ow1 = int(tx * self.ttw)
        ow2 = int(ow1 + self.ttw)

        # print("-------")
        # print(oh1)
        # print(oh2)
        # print(ow1)
        # print(ow2)

        oimg_path = self.params['image_output_dir'] + '/' + \
            padint(gen, 5) + '.jpg'
        print(oimg_path)

        sh1 = int(self.tth * sy)
        sh2 = int(sh1 + self.tth)
        sw1 = int(self.ttw * sx)
        sw2 = int(sw1 + self.ttw)

        # print("=========")
        # print(sh1)
        # print(sh2)
        # print(sw1)
        # print(sw2)
        # oh2 = int(oh1 + min(self.tth, sth))
        # ow2 = int(ow1 + min(self.ttw, stw))
        # print("<<<<<>>>>>")
        # print(oh2)
        # print(ow2)
        try:
            self.overlay_image[oh1:oh2, ow1:ow2, :] = limg[sh1:sh2, sw1:sw2, :]
            self.write_overlay_image(oimg_path)
        except ValueError:
            print("skipping this generation's image overlay write due to ValueError")

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

    def update_for_n_generations(self, gens=10):
        p = self.params
        for gen in range(gens):
            op = 3
            # op = self.choose_op()
            if op == 0:
                # self.modify_scan_test((self.generations + gen))
                pass
            elif op == 1:
                self.modify_rule1((self.generations + gen))
            elif op == 2:
                self.modify_rule2((self.generations + gen))
            elif op == 3:
                gencount = self.modify_rule2a((self.generations + gen))
                self.generations += gencount
            elif op == 3:
                self.modify_rule3((self.generations + gen))
            else:
                pass
            output_path = p['image_output_dir'] + \
                '/' + padint((self.generations + gen), 5) + '.jpg'
            cv.imwrite(output_path, self.overlay_image)
        self.generations += gens
        print("Done! Use ffmpeg to create your animation...")


    # def generate_neigbors_to_check(self, n=1):
    #     p = self.params

    #     rando_choice = random.randint(0, 27)
    #     target_path = p['image_input_dir'] + '/' + padint(rando_choice, 3) + '.jpg'
    #     self.target_image = cv.imread(target_path)
        
    #     rando_tile_r = random.randint(0,9)
    #     rando_tile_c = random.randint(0,9)

    #     # img_histo_path = target_path + '.histo'
        
    #     try:
    #         timg_nn = pickle.loads(open(target_path + '.histo.nn', 'rb').read())
    #     except FileNotFoundError:
    #         print("Skipping bad nn file!")
    #         print(img_histo_path)
    #         return

    #     print(timg_nn.keys())

    #     choice = timg_nn[(rando_choice, float(rando_tile_r), float(rando_tile_c))]

    #     print(choice)


        # output_path = p['image_output_dir'] + '/' + padint(rando_choice, 5) + '.jpg'
        # cv.imwrite(output_path, sim_mtrx)

    # from image_composite_tiles import *
    # ict = ImageCompositeTiles()
    # ict.generate_neigbors_to_check()