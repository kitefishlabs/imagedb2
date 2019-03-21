import os
import color_features as cf
import numpy as np

def test_create():
    feats = cf.ColorFeatures()
    assert(feats.analysis_params['verbose'] == False)
    assert(feats.analysis_params['media_file_path'] == os.path.abspath('img/gradient.png'))

def test_create_with_map_relative_path():
    """
    Should expand path relative to root of source directory.
    """
    feats = cf.ColorFeatures(**{'media_file_path': 'img/foo.jpg'})
    assert(feats.analysis_params['media_file_path'] == os.path.abspath('img/foo.jpg'))

def test_create_with_map_abs_path():
    """
    Should not expand path.
    """
    feats = cf.ColorFeatures(**{'media_file_path': '/home/me/img/foo.jpg'})
    assert(feats.analysis_params['media_file_path'] == '/home/me/img/foo.jpg')

def test_analyze():
    """
    Smoke test; just returns shape of image.
    """
    feats = cf.ColorFeatures()
    shp = feats._analyze_image()
    assert(shp == (720, 1280, 3))

def test_analysis_result():
    """
    Check fp creation and shape by accessing 2 ways. Set bins to 16 explicitly to reinforce that value.
    """
    feats = cf.ColorFeatures(**{'bins':16})
    shp = feats._analyze_image()
    assert(shp == (720, 1280, 3))
    fp_ = feats.get_analysis_result()
    datapath = os.path.abspath('img/gradient.png.lab')
    fp = np.memmap(datapath, dtype='float32', mode='r+', shape=(17, 3, 16))
    assert(fp.shape == (17, 3, 16))
    assert(fp.shape == fp_.shape)

def test_analyze_rgb():
    """
    Test expected (normalized) bin values for image known to have all pixels == RGB(1,144,0).
    """
    feats = cf.ColorFeatures(**{'media_file_path':'img/green.jpg', 'color_space':'rgb'})
    shp = feats._analyze_image()
    assert(shp == (1600, 2560, 3))
    fp_ = feats.get_analysis_result()
    
    assert((sum([h[0] for h in fp_[:,0,:]])) == 17)
    assert((sum([h[9] for h in fp_[:,1,:]])) == 17)
    assert((sum([h[0] for h in fp_[:,2,:]])) == 17)
    
    assert((sum([sum(h) for h in fp_[:,0,:]])) == 17)
    assert((sum([sum(h) for h in fp_[:,1,:]])) == 17)
    assert((sum([sum(h) for h in fp_[:,2,:]])) == 17)
    