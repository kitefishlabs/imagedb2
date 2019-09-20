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

    def __init__(self, **params):
        self.initialize(params)

    def initialize(self, params=None):
        self._check_cf_params(params)
        self.image_tile_index = {}

    def _check_cf_params(self, params=None):
        """
        Simple mechanism to read in default parameters while substituting custom parameters.
        """
        self.params = params if params is not None else self.params
        dcfp = self.default_cf_params()
        for k in dcfp.keys():
            if k == 'media_directory':
                self.params[k] = os.path.abspath(
                    self.params.get(k, dcfp[k]))
            else:
                self.params[k] = self.params.get(k, dcfp[k])
        return self.params

    @staticmethod
    def default_cf_params():
        params = {
            # default dir
            'media_directory': os.path.abspath('img')
        }
        return params

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
        for filepath in glob.glob(self.params['media_directory'] + '/*.jpg'):
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

    def extract_file_id(self, fpath):
        idnum = -1
        m = re.search(r'(\d+).jpg', path.basename(fpath))
        if m is not None:
            idnum = m.group(1)
        return int(idnum)

    def index_connections(self):
        """
        Internal similarities are expected to be prevalent in general, and should also correlate to positional proximity. Starting with option C.
        A/ Rank similar but distant tiles higher. (more likely to spawn images with internal consistency/repetition?)
        B/ Rank similar and proximate tiles higher (less churn, repetition?)
        C/ Ignore distance
        """
        all_files = glob.glob(
            self.params['media_directory'] + '/*.jpg.histo')
        all_files_ = glob.glob(
            self.params['media_directory/hide'] + '/*.jpg.histo')

        target_slots = {}

        for filepatha in all_files:

            a = self.extract_file_id(filepatha)
            indexa = pickle.loads(open(filepatha, "rb").read())
            rankings = {}

            for filepathb in all_files_:

                # if b >= a:
                b = self.extract_file_id(filepathb)
                indexb = pickle.loads(open(filepathb, "rb").read())

                for cell in indexa:

                    scores = []
                    val = (self.convert_8(cell[0]), self.convert_8(
                        cell[1]), self.convert_8(cell[2]))

                    for cell2 in indexb:

                        val2 = (self.convert_8(cell2[0]), self.convert_8(
                            cell2[1]), self.convert_8(cell2[2]))
                        # print((i, j, self.convert_8(cell[0]), self.convert_8(
                        # cell[1]), self.convert_8(cell[2]), self.convert_8(cell2[0]), self.convert_8(
                        # cell2[1]), self.convert_8(cell2[2])))
                        diff0 = self.diff_histograms(val[0], val2[0])
                        diff1 = self.diff_histograms(val[1], val2[1])
                        diff2 = self.diff_histograms(val[2], val2[2])
                        scores += [[a, b, cell[4], cell[5], cell2[4], cell2[5],
                                    (diff0 + diff1 + diff2)]]
                        try:
                            existing = target_slots[(a, b)]
                            target_slots[(a, b)] = (min(existing[0], cell2[4]),
                                                    #   max(existing[0], a[4]),
                                                    #   min(existing[1], x[4]),
                                                    max(existing[1], cell2[4]),
                                                    min(existing[2], cell2[5]),
                                                    #   max(existing[2], a[5]),
                                                    #   min(existing[3], a[5]),
                                                    max(existing[3], cell2[5]))
                        except KeyError:
                            target_slots[(a, b)] = (
                                cell2[4], cell2[4], cell2[5], cell2[5])
                    # each cell is the distances to all 256 other cells (including itself)
                    # now sort the cell's connections
                    # print(scores)
                    sorted_by_dist = sorted(
                        scores, key=(lambda score: score[-1]))[:20]
                    rankings[(b, cell[4], cell[5])] = sorted_by_dist
            # print("TARG:")
            # print(target_slots)

            f = open(filepatha + ".nn", "wb")
            f.write(pickle.dumps(rankings))
            f.close()
        return target_slots

    # In order to choose the files first and then within the files, we need to index each target file/point -> averaged distance combo so we can order by similarity over the entire set of points for each file. That way we shouldn't have "least bad choices" from minimizing the dist. func after a distant(dissimilar) file choice. A cursory glance at the data suggest that each individual file's points in similarity space may be clustered when mapped onto larger search space.

    def index_files(self):
        all_files = glob.glob(
            self.params['media_directory'] + '/*.jpg.histo.nn')

        all_sorted_dists = {}
        for filepatha in all_files:

            a = self.extract_file_id(filepatha)
            indexa = pickle.loads(open(filepatha, "rb").read())
            general_dist_ranking = []

            for key in indexa:
                # print(key) # to-id, from-c, from-r, to-c, to-r
                total = 0.0
                for item in indexa[key]:
                    assert(len(item) == 7)
                    total += item[-1]
                assert(key[1] == item[-5])
                assert(key[2] == item[-4])
                # print(indexa[key])
                # print((total, (key[0], key[1], key[2], item[-3], item[-2])))
                general_dist_ranking += [(total, (key[0],
                                                  key[1], key[2], item[-3], item[-2]))]

            dist_ordered = sorted(general_dist_ranking, key=(lambda dm: dm[0]))

            dist_ = [item[0] for item in dist_ordered]
            payload_ = [item[1] for item in dist_ordered]
            dist_ /= np.max(np.abs(dist_), axis=0)
            dlen = len(dist_)
            dist_ = [((1. - x) + ((((1. - (i / dlen)) * 10.) ** 2.) / 100.))
                     for i, x in enumerate(dist_)]
            combined = list(zip(dist_, payload_))

            fileid = self.extract_file_id(filepatha)
            all_sorted_dists[fileid] = (dist_, combined)

            # print('------>>>>>>')
            # print()
            # print(combined)
            # print()
            # print()

            f = open(filepatha + "2", "wb")
            f.write(pickle.dumps(combined))
            f.close()

        return all_sorted_dists

    def plumb_range(self, id):

        limg_path = self.params['media_directory'] + \
            '/' + padint(id, 3) + '.jpg'
        limg = cv.imread(limg_path)
        if limg is not None:
            print(limg.shape)
        limg_histo_path = self.params['media_directory'] + \
            '/' + padint(id, 3) + '.jpg.histo'

        try:
            limg_histo = pickle.loads(open(limg_histo_path, 'rb').read())
        except FileNotFoundError:
            print("NotFound")
            return

        target_slots = {}
        for x in limg_histo:
            print(x)
            try:
                existing = target_slots[x[3]]
                target_slots[x[3]] = (min(existing[0], x[4]),
                                      #   max(existing[0], a[4]),
                                      #   min(existing[1], x[4]),
                                      max(existing[1], x[4]),
                                      min(existing[2], x[5]),
                                      #   max(existing[2], a[5]),
                                      #   min(existing[3], a[5]),
                                      max(existing[3], x[5]))
            except KeyError:
                target_slots[x[3]] = (x[4], x[4], x[5], x[5])
        return target_slots


# idx = ith.index_files()
# distrib = sorted(list(set([lst[0] for lst in idx])))
# plt.plot(distrib)
