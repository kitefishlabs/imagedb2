import color_features_pyramid as cfp
import matplotlib.pyplot as plt
import numpy as np

# We could have taken the defaults, too
mycfp = cfp.ColorFeaturesPyramid(**{'media_file_path': "/Users/kfl/dev/git/imagedb2/img/001.jpg", 'down_sampling': 8, 'tilew': 16, 'tileh': 12})

lvl = 6
#
data = list(mycfp._down_sample_and_analyze())
print(len(data))
# down-sampling 2 ** 3 = 8, r, g, b (idx 1, 2, 3 of data tuple)
data_ = np.concatenate([data[lvl][1][:,:16].T, data[lvl][2][:,:16].T, data[lvl][3][:,:16].T])  
data_len = data[3][1].shape[0]
# This will plot the first 1600 frames analyzed within an imagedatetime A combination of a date and a time. It's a rough sketch of the data.

a1 = plt.subplot2grid((4,4),(0,0),colspan=4)
a2 = plt.subplot2grid((4,4),(1,0),colspan=4)
a3 = plt.subplot2grid((4,4),(2,0),colspan=4)
a4 = plt.subplot2grid((4,4),(3,0),colspan=4)
oneq = int(data_len / 4)
half = int(data_len / 2)
threeq = int(data_len / 4 * 3)
a1.imshow(data_[:16,:oneq], cmap='hot', interpolation='nearest')
a2.imshow(data_[:16,oneq:half], cmap='hot', interpolation='nearest')
a3.imshow(data_[:16,half:threeq], cmap='hot', interpolation='nearest')
a4.imshow(data_[:16,threeq:], cmap='hot', interpolation='nearest')
plt.show()
