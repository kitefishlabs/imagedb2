import unittest
import os
import image_tiles_hash as ith
import numpy as np

class Tests(unittest.TestCase):
    def test_create(self):
        test_ith = ith.ImageTilesHash()
        self.assertEqual(test_ith.params['media_directory'], os.path.abspath('img'))
        self.assertEqual(test_ith.image_index, {})
        self.assertEqual(test_ith.image_tile_index, {})

if __name__ == '__main__':
    unittest.main()
