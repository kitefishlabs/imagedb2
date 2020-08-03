import unittest
import os
import color_features_pyramid as cfp
import numpy as np

class Tests(unittest.TestCase):
  def test_create(self):
    test_icp = cfp.ColorFeaturesPyramid()
    self.assertIsNotNone(test_icp)

  def test_default_path(self):
    test_cfp = cfp.ColorFeaturesPyramid()
    self.assertEqual(test_cfp.params['media_file_path'], os.path.abspath('img/001.jpg'))

  def test_default_color_space(self):
    test_cfp = cfp.ColorFeaturesPyramid()
    self.assertEqual(test_cfp.params['color_space'], 'lab')

  def test_determine_num_bins(self):
    test_cfp = cfp.ColorFeaturesPyramid()
    self.assertEqual(test_cfp._determine_num_bins(960, 2), 120)
    self.assertEqual(test_cfp._determine_num_bins(960, 8), 1)
    self.assertEqual(test_cfp._determine_num_bins(960, 9), 0)
  
  def test_zeros_for_shape_level(self):
    test_cfp = cfp.ColorFeaturesPyramid()
    self.assertEqual(test_cfp._zeros_for_shape_level((960, 720), 0).shape, (360, 480, 3))
    self.assertEqual(test_cfp._zeros_for_shape_level((960, 720), 1).shape, (180, 240, 3))
    self.assertEqual(test_cfp._zeros_for_shape_level((960, 720), 2).shape, (90, 120, 3))
  
  def test_zeros_for_shape_level_limit(self):
    test_cfp = cfp.ColorFeaturesPyramid()
    self.assertEqual(test_cfp._zeros_for_shape_level((960, 720), 0).shape, (360, 480, 3))
    self.assertEqual(test_cfp._zeros_for_shape_level((960, 720), 8).shape, (1, 1, 3))
    self.assertEqual(test_cfp._zeros_for_shape_level((960, 720), 9).shape, (0, 0, 3))
  
  # def test_up_sample_in_place(self):
  #   test_cfp = cfp.ColorFeaturesPyramid()
  #   test_in_mtrx = np.array([[r0,g0,b0], ])
  #   self.assertEqual(test_cfp._down_sample_in_place(), ) # in_mtrx, pwr):


if __name__ == '__main__':
  unittest.main()
