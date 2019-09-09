from image_tiles_hash import *
from image_composite_tiles import *
ict = ImageCompositeTiles(**{'target_id': 8})
ict.init_images_and_data(0)
ict.update_for_n_generations(1)

#(ict.numh * ict.numw)


ith = ImageTilesHash()
res = ith.index_files()
ts = ith.plumb_range(res)

limg_histo_path = self.params['image_input_dir'] + \
    '/' + self.padint(id, 3) + '.jpg.histo'
try:
    limg_histo = pickle.loads(open(limg_histo_path, 'rb').read())
except FileNotFoundError:
    print("NotFound")
