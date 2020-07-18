import color_features_tiles as cft
import matplotlib.pyplot as plt
import numpy as np
# We could have taken the defaults, too
mycft = cft.ColorFeaturesTiles(**{'media_file_path': "/Users/kfl/dev/git/imagedb2/img/001.jpg", 'tile_divs': (2 ** 4)})

# All three channels
d0 = mycft._sample_and_analyze(0)
d1 = mycft._sample_and_analyze(1)
d2 = mycft._sample_and_analyze(2)

d_all = np.concatenate([d0[:,:16].T, d1[:,:16].T, d2[:,:16].T]) 

# This will plot the first 1600 frames analyzed within an imagedatetime A combination of a date and a time. It's a rough sketch of the data.

a1 = plt.subplot2grid((4,4),(0,0),colspan=4)
a2 = plt.subplot2grid((4,4),(1,0),colspan=4)
a3 = plt.subplot2grid((4,4),(2,0),colspan=4)
a4 = plt.subplot2grid((4,4),(3,0),colspan=4)
a1.imshow(d_all[:48,:400], cmap='hot', interpolation='nearest')
a2.imshow(d_all[:48,400:800], cmap='hot', interpolation='nearest')
a3.imshow(d_all[:48,800:1200], cmap='hot', interpolation='nearest')
a4.imshow(d_all[:48,1200:1600], cmap='hot', interpolation='nearest')
plt.show()
