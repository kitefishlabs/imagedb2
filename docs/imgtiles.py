import color_features_tiles as cft
import matplotlib.pyplot as plt                                                                                                                     
# We could have taken the defaults, too
mycft = cft.ColorFeaturesTiles(**{'media_file_path': "/Users/kfl/dev/git/imagedb2/img/000.jpg", 'tile_divs': (2 ** 4)})

# Just the first channel!
d0 = mycft._sample_and_analyze(0) 

# This will plot the first 1600 frames analyzed within an imagedatetime A combination of a date and a time. It's a rough sketch of the data.

a1 = plt.subplot2grid((4,4),(0,0),colspan=4)
a2 = plt.subplot2grid((4,4),(1,0),colspan=4)
a3 = plt.subplot2grid((4,4),(2,0),colspan=4)
a4 = plt.subplot2grid((4,4),(3,0),colspan=4)
a1.imshow(d0.T[:16,:400], cmap='hot', interpolation='nearest')
a2.imshow(d0.T[:16,400:800], cmap='hot', interpolation='nearest')
a3.imshow(d0.T[:16,800:1200], cmap='hot', interpolation='nearest')
a4.imshow(d0.T[:16,1200:1600], cmap='hot', interpolation='nearest')
plt.show()
