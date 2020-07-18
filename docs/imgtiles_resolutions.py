import color_features_tiles as cft
import matplotlib.pyplot as plt
import numpy as np

# Three instance, with different resolutions
mycft1 = cft.ColorFeaturesTiles(**{'media_file_path': "/Users/kfl/dev/git/imagedb2/img/001.jpg", 'tile_divs': (2 ** 4), 'tilew': 64, 'tileh': 48})

# All three channels
d10 = mycft1._sample_and_analyze(0)
d11 = mycft1._sample_and_analyze(1)
d12 = mycft1._sample_and_analyze(2)

d_all1 = np.concatenate([d10[:,:16].T, d11[:,:16].T, d12[:,:16].T]) 

# Three instance, with different resolutions
mycft2 = cft.ColorFeaturesTiles(**{'media_file_path': "/Users/kfl/dev/git/imagedb2/img/001.jpg", 'tile_divs': (2 ** 4), 'tilew': 32, 'tileh': 24})

# All three channels
d20 = mycft2._sample_and_analyze(0)
d21 = mycft2._sample_and_analyze(1)
d22 = mycft2._sample_and_analyze(2)

d_all2 = np.concatenate([d20[:,:16].T, d21[:,:16].T, d22[:,:16].T]) 

# Three instance, with different resolutions
mycft3 = cft.ColorFeaturesTiles(**{'media_file_path': "/Users/kfl/dev/git/imagedb2/img/001.jpg", 'tile_divs': (2 ** 4), 'tilew': 16, 'tileh': 12})

# All three channels
d30 = mycft3._sample_and_analyze(0)
d31 = mycft3._sample_and_analyze(1)
d32 = mycft3._sample_and_analyze(2)

d_all3 = np.concatenate([d30[:,:16].T, d31[:,:16].T, d32[:,:16].T]) 


# This will plot the first 1600 frames analyzed within an imagedatetime A combination of a date and a time. It's a rough sketch of the data.

a1 = plt.subplot2grid((4,4),(0,0),colspan=4)
a2 = plt.subplot2grid((4,4),(1,0),colspan=4)
a3 = plt.subplot2grid((4,4),(2,0),colspan=4)
a4 = plt.subplot2grid((4,4),(3,0),colspan=4)
a1.imshow(d_all1[:48,:600], cmap='hot', interpolation='nearest')
a2.imshow(d_all2[:48,:600], cmap='hot', interpolation='nearest')
a3.imshow(d_all3[:48,:600], cmap='hot', interpolation='nearest')
a4.imshow(d_all3[:48,600:1200], cmap='hot', interpolation='nearest')
plt.show()
