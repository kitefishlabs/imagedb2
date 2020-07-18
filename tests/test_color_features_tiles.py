import unittest
import os
import color_features_tiles as cft
import numpy as np

class Tests(unittest.TestCase):
  def test_create(self):
    test_ict = cft.ColorFeaturesTiles()
    self.assertIsNotNone(test_ict)

  def test_default_path(self):
    test_ict = cft.ColorFeaturesTiles()
    self.assertEqual(test_ict.params['media_file_path'], os.path.abspath('img/000.jpg'))

  def test_default_color_space(self):
    test_ict = cft.ColorFeaturesTiles()
    self.assertEqual(test_ict.params['color_space'], 'lab')

if __name__ == '__main__':
  unittest.main()
