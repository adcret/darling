import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from darling.assets import energy_scan, gaussian_blobs, mosaicity_scan


class TestAssets(unittest.TestCase):
    # Tests for the darling.assets module.

    def setUp(self):
        self.debug = False

    def test_mosaicity_scan(self):
        path, data, coordinates = mosaicity_scan()

        self.assertTrue(isinstance(path, str))
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(data.shape[2] == len(coordinates[0]))
        self.assertTrue(data.shape[3] == len(coordinates[1]))
        self.assertTrue(coordinates[0].dtype == np.float32)
        self.assertTrue(coordinates[1].dtype == np.float32)
        self.assertTrue(len(coordinates[0].shape) == 1)
        self.assertTrue(len(coordinates[1].shape) == 1)

    def test_energy_scan(self):
        path, data, coordinates = energy_scan()

        self.assertTrue(isinstance(path, str))
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(data.shape[2] == len(coordinates[0]))
        self.assertTrue(data.shape[3] == len(coordinates[1]))
        self.assertTrue(coordinates[0].dtype == np.float32)
        self.assertTrue(coordinates[1].dtype == np.float32)
        self.assertTrue(len(coordinates[0].shape) == 1)
        self.assertTrue(len(coordinates[1].shape) == 1)

    def test_gaussian_blobs(self):
        m = 5
        N = 19
        data, coordinates = gaussian_blobs(N=N, m=m)

        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(data.shape[2] == m)
        self.assertTrue(data.shape[3] == m)
        self.assertTrue(len(coordinates[0].shape) == 1)
        self.assertTrue(len(coordinates[1].shape) == 1)


if __name__ == "__main__":
    unittest.main()