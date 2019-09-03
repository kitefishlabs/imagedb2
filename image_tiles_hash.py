import glob
import pickle
import random
from color_features_tiles import *
import numpy as np
from utils import *


class ImageTilesHash:
    """
    The aim is to create a hash table mapping color histogram data to tiles in the image.
    """

    def __init__(self, **analysis_params):
        self.initialize(analysis_params)

    def initialize(self, analysis_params=None):
        self._check_cf_params(analysis_params)
        self.image_tile_index = {}

    def _check_cf_params(self, analysis_params=None):
        """
        Simple mechanism to read in default parameters while substituting custom parameters.
        """
        self.analysis_params = analysis_params if analysis_params is not None else self.analysis_params
        dcfp = self.default_cf_params()
        for k in dcfp.keys():
            if k == 'media_directory':
                self.analysis_params[k] = os.path.abspath(
                    self.analysis_params.get(k, dcfp[k]))
            else:
                self.analysis_params[k] = self.analysis_params.get(k, dcfp[k])
        return self.analysis_params

    @staticmethod
    def default_cf_params():
        analysis_params = {
            # default dir
            'media_directory': os.path.abspath('img')
        }
        return analysis_params

    def build_index_for_image(self, path, exp=4):

        bins = 2 ** exp

        cft = ColorFeaturesTiles(
            **{'media_file_path': path, 'tile_divs': bins})
        data0 = cft._sample_and_analyze(0)
        data1 = cft._sample_and_analyze(1)
        data2 = cft._sample_and_analyze(2)

        computed0 = []
        for row in data0:
            computed = 0
            for i, v in enumerate(list(row[:-5])):
                computed += int(v * bins) << (i * exp)
                # print("c0/ ", i, v, (int(v * bins) << (i * exp)), "\n")
            computed0 += [computed]

        computed1 = []
        for row in data1:
            computed = 0
            for i, v in enumerate(list(row[:-5])):
                computed += int(v * bins) << (i * exp)
                # print("c1/ ", i, v, (int(v * bins) << (i * exp)), "\n")
            computed1 += [computed]

        computed2 = []
        for row in data2:
            computed = 0
            for i, v in enumerate(list(row[:-5])):
                computed += int(v * bins) << (i * exp)
                # print("c2/ ", i, v, (int(v * bins) << (i * exp)), "\n")
            computed2 += [computed]

        infos_by_row = [row[-5:] for row in data0]

        trios = []
        for i in range(len(computed0)):
            info = infos_by_row[i]
            trios.append((computed0[i], computed1[i], computed2[i], int(
                info[0]), info[1], info[2], info[3], info[4]))

        return trios

    def index_directory(self):
        for filepath in glob.glob(self.analysis_params['media_directory'] + '/*.jpg'):
            res = self.build_index_for_image(filepath, 4)
            f = open(filepath + ".histo", "wb")
            f.write(pickle.dumps(res))
            f.close()

    def convert_8(self, val):
        res = [(((val >> (i*3)) % 16) / 16.0) for i in range(16)]
        res.reverse()
        return(res)

    def diff_histograms(self, hista, histb):
        assert(len(hista) == len(histb))
        total = 0
        for a, b in zip(hista, histb):
            total += ((a - b) ** 2.0)
            # total += abs(a - b)
        return total / len(hista)

    def index_internal_connections(self):
        """
        Internal similarities are expected to be prevalent in general, and should also correlate to positional proximity. Starting with option C.
        A/ Rank similar but distant tiles higher. (more likely to spawn images with internal consistency/repetition?)
        B/ Rank similar and proximate tiles higher (less churn, repetition?)
        C/ Ignore distance
        """
        for filepath in glob.glob(self.analysis_params['media_directory'] + '/*.jpg.histo'):
            # idnum = -1
            # m = re.search(r'(\d+).[jpg|png]', path.basename(filepath))
            # if m is not None:
            #     idnum = m.group(1)
            index = pickle.loads(open(filepath, "rb").read())
            rankings = []
            for (i, cell) in enumerate(index):
                scores = []
                val = (self.convert_8(cell[0]), self.convert_8(
                    cell[1]), self.convert_8(cell[2]))
                for (j, cell2) in enumerate(index):
                    if ((j >= i) and (i != j)):
                        val2 = (self.convert_8(cell2[0]), self.convert_8(
                            cell2[1]), self.convert_8(cell2[2]))
                        # print((i, j, self.convert_8(cell[0]), self.convert_8(
                        # cell[1]), self.convert_8(cell[2]), self.convert_8(cell2[0]), self.convert_8(
                        # cell2[1]), self.convert_8(cell2[2])))
                        diff0 = self.diff_histograms(val[0], val2[0])
                        diff1 = self.diff_histograms(val[1], val2[1])
                        diff2 = self.diff_histograms(val[2], val2[2])
                        scores += [(i, j, (diff0 + diff1 + diff2))]
                # each cell is the distances to all 256 other cells (including itself)
                # now sort the cell's connections
                print(scores)
                sorted_by_dist = sorted(
                    scores, key=(lambda score: score[2]))[:20]
                rankings += sorted_by_dist
            # f = open(filepath + ".internal", "wb")
            # f.write(pickle.dumps(res))
            # f.close()
            return rankings
