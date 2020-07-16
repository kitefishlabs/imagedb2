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
        self.image_index = {}
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
        # there are 3 color planes to work with
        # they are analyzed separately
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

    def index_directory(self, guard=True):
        """
        index each image by color hash in L*a*b* color space.
        """
        for filepath in glob.glob(self.params['media_directory'] + '/*.jpg'):
            if (guard and path.exists(filepath + ".histo")):
                pass
            else:
                res = self.build_index_for_image(filepath, 4)
                f = open(filepath + ".histo", "wb")
                f.write(pickle.dumps(res))
                f.close()

    def convert_8(self, val, div):
        res = [(((val >> (i*3)) % div) / float(div)) for i in range(div)]
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

    def index_internal_connections(self):
        """
        """
        all_files = glob.glob(
            self.params['media_directory'] + '/*.jpg.histo')

        # target_slots = {}

        for filepatha in all_files:

            rankings = {}
            a = self.extract_file_id(filepatha)
            b = a
            indexa = pickle.loads(open(filepatha, "rb").read())
            indexb = indexa[:]  # pickle.loads(open(filepathb, "rb").read())

            for cell in indexa:

                scores = []
                valL = self.convert_8(cell[0], 16)
                vala = self.convert_8(cell[1], 16)
                valb = self.convert_8(cell[2], 16)

                for cell2 in indexb:

                    val2L = self.convert_8(cell2[0], 16)
                    val2a = self.convert_8(cell2[1], 16)
                    val2b = self.convert_8(cell2[2], 16)

                    # diffs on each: L * a * b
                    diff0 = self.diff_histograms(valL, val2L)
                    diff1 = self.diff_histograms(vala, val2a)
                    diff2 = self.diff_histograms(valb, val2b)

                    # score is average of diffs
                    scores += [[a, b, cell[4], cell[5], cell2[4], cell2[5],
                                ((diff0 + diff1 + diff2) / 3.0)]]

                    # try:
                    #     existing = target_slots[(a, b)]
                    #     target_slots[(a, b)] = (min(existing[0], cell2[4]),
                    #                             #   max(existing[0], a[4]),
                    #                             #   min(existing[1], x[4]),
                    #                             max(existing[1], cell2[4]),
                    #                             min(existing[2], cell2[5]),
                    #                             #   max(existing[2], a[5]),
                    #                             #   min(existing[3], a[5]),
                    #                             max(existing[3], cell2[5]))
                    # except KeyError:
                    #     target_slots[(a, b)] = (
                    #         cell2[4], cell2[4], cell2[5], cell2[5])
                # each cell is the distances to all 256 other cells (including itself)
                # now sort the cell's connections
                sorted_by_dist = sorted(
                    scores, key=(lambda score: score[-1]))[:100]
                rankings[(b, cell[4], cell[5])] = sorted_by_dist

                # print("TARG:")
                # print(target_slots)

            f = open(filepatha + ".nn1", "wb")
            f.write(pickle.dumps(rankings))
            f.close()
        return target_slots

    def index_connections(self, guard=True):
        """
        Internal similarities are expected to be prevalent in general, and should also correlate to positional proximity. Starting with option C.
        A/ Rank similar but distant tiles higher. (more likely to spawn images with internal consistency/repetition?)
        B/ Rank similar and proximate tiles higher (less churn, repetition?)
        C/ Ignore distance
        """
        all_files = glob.glob(
            self.params['media_directory'] + '/*.jpg.histo')
        all_files_ = glob.glob(
            self.params['media_directory'] + '/*.jpg.histo')

        # pickled_files = {}
        # target_slots = {}

        for filepatha in all_files:

            if (guard and path.exists(filepatha + ".nn")):
                pass

            else:
                rankings = {}
                a = self.extract_file_id(filepatha)

                # try:
                # indexa = pickled_files[filepatha]
                # except KeyError:
                indexa = pickle.loads(open(filepatha, "rb").read())
                # pickled_files[filepatha] = indexa

                for filepathb in all_files_:

                    b = self.extract_file_id(filepathb)
                    if b >= a:

                        # try:
                        # indexb = pickled_files[filepathb]
                        # except KeyError:
                        indexb = pickle.loads(open(filepathb, "rb").read())
                        # pickled_files[filepathb] = indexb

                        for cell in indexa:

                            scoresa = []
                            scoresb = []
                            valL = self.convert_8(cell[0], 16)
                            vala = self.convert_8(cell[1], 16)
                            valb = self.convert_8(cell[2], 16)

                            for cell2 in indexb:

                                val2L = self.convert_8(cell2[0], 16)
                                val2a = self.convert_8(cell2[1], 16)
                                val2b = self.convert_8(cell2[2], 16)

                                # diffs on each: L * a * b
                                diff0 = self.diff_histograms(valL, val2L)
                                diff1 = self.diff_histograms(vala, val2a)
                                diff2 = self.diff_histograms(valb, val2b)

                                score = (diff0 + diff1 + diff2) / 3.0
                                # score is average of diffs
                                scoresa += [[a, b, cell[4], cell[5],
                                             cell2[4], cell2[5], score]]
                                scoresb += [[b, a, cell2[4], cell2[5],
                                             cell[4], cell[5], score]]
                                # try:
                                #     existing = target_slots[(a, b)]
                                #     target_slots[(a, b)] = (min(existing[0], cell2[4]),
                                #                             #   max(existing[0], a[4]),
                                #                             #   min(existing[1], x[4]),
                                #                             max(existing[1], cell2[4]),
                                #                             min(existing[2], cell2[5]),
                                #                             #   max(existing[2], a[5]),
                                #                             #   min(existing[3], a[5]),
                                #                             max(existing[3], cell2[5]))
                                # except KeyError:
                                #     target_slots[(a, b)] = (
                                #         cell2[4], cell2[4], cell2[5], cell2[5])
                            # each cell is the distances to all 256 other cells (including itself)
                            # now sort the cell's connections
                            sorted_by_dista = sorted(
                                scoresa, key=(lambda score: score[-1]))[:100]
                            sorted_by_distb = sorted(
                                scoresb, key=(lambda score: score[-1]))[:100]
                            rankings[(b, cell[4], cell[5])] = sorted_by_dista
                            rankings[(a, cell[4], cell[5])] = sorted_by_distb
                    # print("TARG:")
                    # print(target_slots)

                f = open(filepatha + ".nn", "wb")
                f.write(pickle.dumps(rankings))
                f.close()
        return target_slots

    def load_connections(self, fileid):
        filepath = self.params['media_directory'] + \
            '/' + padint(fileid, 3) + '.jpg.histo.nn'
        return pickle.loads(open(filepath, "rb").read())

    def load_all_connections(self):
        all_connections = {}
        for i in range(27):
            all_connections.update(load_connections(i))
        return all_connections

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
                # print(key)  # to-id, from-c, from-r
                total = 0.0
                alen = len(indexa[key])
                for item in indexa[key]:
                    assert(len(item) == 7)
                    total += item[-1]
                    # print((item[-3], item[-2]))
                if total > 0.0:
                    total /= alen
                assert(key[1] == item[-5])
                assert(key[2] == item[-4])
                # print(alen)
                # print((total, (key[0], key[1], key[2])))
                general_dist_ranking += [(total, (key[0],
                                                  key[1], key[2]))]

            dist_ordered = sorted(general_dist_ranking, key=(
                lambda dm: dm[0]))

            dist_ = [item[0] for item in dist_ordered]
            payload_ = [item[1] for item in dist_ordered]
            dist_ /= np.max(np.abs(dist_), axis=0)
            # dlen = len(dist_)
            # dist_ = [((1. - x) + ((((1. - (i / dlen)) * 10.) ** 2.) / 100.))
            #  for i, x in enumerate(dist_)]
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

    def load_image_tile_index(self):
        """
        Loads all image tile histogram nn data.
        source file id is implied/gathered from each loaded files' name/id.
        per indexed file:
            HISTO.NN: (target_fileid, source_row, source_col) -> (distance, (to-id, from-r, from-c, to-r, to-c))
            expand each key -> (source_fileid, target_fileid, source_row, source_col) -> same payloads
        store map of expanded source mappings for queries as self.image_tile_index
        """
        all_files = glob.glob(
            self.params['media_directory'] + '/*.jpg.histo.nn')
        self.image_tile_index = {}
        for filepatha in all_files:
            a = self.extract_file_id(filepatha)
            expanded_indexa = {}
            indexa = pickle.loads(open(filepatha, "rb").read())
            for key in indexa:
                expanded_indexa[(a, key[0], key[1], key[2])] = indexa[key]
            self.image_tile_index.update(expanded_indexa)
        return self.image_tile_index

    def load_image_index(self, srcfileid):
        """
        Loads all image-index data.
        per indexed file:
            HISTO.NN2: (distance -> (target_field,source_row,source_col))
        """
        try:
            self.image_index[srcfileid]
        except KeyError:
            filepatha = self.params['media_directory'] + \
                '/' + padint(srcfileid, 3) + '.jpg.histo.nn2'
            expanded_indexa = []
            # indexa here is connections sorted by distance
            indexa = pickle.loads(open(filepatha, "rb").read())
            for item in indexa:
                expanded_indexa += [(item[0], (srcfileid,
                                               item[1][0], item[1][1], item[1][2]))]
            self.image_index[srcfileid] = expanded_indexa
        return self.image_index

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

    def lookup_image_index(self, a):
        try:
            return self.image_index[a]
        except KeyError:
            print("\n\nloading image index...\n\n")
            return self.load_image_index(a)[a]

    def lookup_image_tile_index(self, a, a_ra_ca_b):
        try:
            return self.image_tile_index[a_ra_ca_b]
        except KeyError:
            print("\n\nloading image tile index...\n\n")
            return self.load_image_tile_index()[a_ra_ca_b]

    def combine_nn_indexes(self):
        all_files = glob.glob(self.params['media_directory'] + '/*.nn')
        nn_index = {}
        for nnfilepath in all_files:
            indexnn = pickle.loads(open(nnfilepath, "rb").read())
            nn_index.update(indexnn)
        for n in range(2):
            f = open(self.params['media_directory'] +
                     "/" + padint(n, 3) + ".jpg.histo.nnc", "wb")
            combined = dict(
                filter(lambda kelem: kelem[0] == n, nn_index.items()))
            f.write(pickle.dumps(combined))
            f.close()
