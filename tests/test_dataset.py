import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import darling


class TestDataSet(unittest.TestCase):
    # Tests for the darling.assets module.

    def setUp(self):
        self.debug = False

        # we test for the mosa scan reader
        path_to_data_1, _, _ = darling.assets.mosaicity_scan()
        self.reader_1 = darling.reader.MosaScan(path_to_data_1)
        self.data_name_1 = "instrument/pco_ff/image"

        # as well as the energy scan reader
        path_to_data_2, _, _ = darling.assets.energy_scan()
        self.reader_2 = darling.reader.EnergyScan(path_to_data_2)

        self.readers = [self.reader_1, self.reader_2]

    def test_init(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)

    def test_load_scan(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)

            # test that a scan can be loaded.
            dset.load_scan(scan_id="1.1", roi=None)
            self.check_data(dset)

            # test the tuple args option
            dset.load_scan(scan_id="1.1", roi=None)
            self.check_data(dset)
            data_layer_1 = dset.data.copy()

            # test to load a diffrent layer
            dset.load_scan(scan_id="2.1", roi=None)
            self.check_data(dset)

            # ensure the data shape is consistent between layers
            self.assertEqual(data_layer_1.shape, dset.data.shape)

            # ensure the data is actually different between layers.
            residual = data_layer_1 - dset.data
            self.assertNotEqual(np.max(np.abs(residual)), 0)

            # test that a roi can be loaded and that the resulting shape is ok.
            dset.load_scan(scan_id="2.1", roi=(0, 9, 3, 19))
            self.check_data(dset)
            self.assertTrue(dset.data.shape[0] == 9)
            self.assertTrue(dset.data.shape[1] == 16)

    def test_subtract(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)
            mm = np.max(dset.data)
            dset.subtract(value=200)
            self.assertEqual(np.max(dset.data), mm - 200)

    def test_moments(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)
            mean, covariance = dset.moments()
            self.assertEqual(mean.shape[0], dset.data.shape[0])
            self.assertEqual(mean.shape[1], dset.data.shape[1])
            self.assertEqual(mean.shape[2], 2)
            self.assertEqual(covariance.shape[0], dset.data.shape[0])
            self.assertEqual(covariance.shape[1], dset.data.shape[1])
            self.assertEqual(covariance.shape[2], 2)
            self.assertEqual(covariance.shape[3], 2)

            if self.debug:
                fig, ax = dset.plot.mean()
                plt.show()

    def test_estimate_mask(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)
            mask = dset.estimate_mask()
            self.assertEqual(mask.shape[0], dset.data.shape[0])
            self.assertEqual(mask.shape[1], dset.data.shape[1])
            self.assertEqual(mask.dtype, bool)

            if self.debug:
                plt.style.use("dark_background")
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                im = ax.imshow(mask)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show()

    def test_integrate(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)
            int_frames = dset.integrate()
            self.assertEqual(int_frames.shape[0], dset.data.shape[0])
            self.assertEqual(int_frames.shape[1], dset.data.shape[1])
            self.assertEqual(int_frames.dtype, np.float32)

            if self.debug:
                plt.style.use("dark_background")
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                im = ax.imshow(int_frames)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show()

    def test_compile_layers(self):
        # test that the mosa reader will stack layers
        path_to_data, _, _ = darling.assets.mosaicity_scan()
        reader = darling.reader.MosaScan(path_to_data)

        dset_mosa = darling.DataSet(reader)

        mean_3d, cov_3d = dset_mosa.compile_layers(
            scan_ids=["1.1", "2.1"], verbose=False
        )

        self.assertEqual(mean_3d.shape[0], 2)
        self.assertEqual(len(mean_3d.shape), 4)
        self.assertEqual(cov_3d.shape[0], 2)
        self.assertEqual(len(cov_3d.shape), 5)

        # test that the energy reader can stack layers.
        path_to_data, _, _ = darling.assets.energy_scan()
        reader = darling.reader.EnergyScan(path_to_data)
        dset_energy = darling.DataSet(reader)
        mean_3d, cov_3d = dset_energy.compile_layers(
            scan_ids=["1.1", "2.1"], verbose=False
        )

        self.assertEqual(mean_3d.shape[0], 2)
        self.assertEqual(len(mean_3d.shape), 4)
        self.assertEqual(cov_3d.shape[0], 2)
        self.assertEqual(len(cov_3d.shape), 5)

    def test_as_paraview(self):
        path_to_data, _, _ = darling.assets.mosaicity_scan()
        reader = darling.reader.MosaScan(path_to_data)

        dset_mosa = darling.DataSet(reader)

        mean_3d, cov_3d = dset_mosa.compile_layers(
            scan_ids=["1.1", "2.1"], verbose=False
        )

        filename = os.path.join(darling.assets.path(), "saves", "mosa_stack")
        dset_mosa.to_paraview(filename)

        path_to_data, _, _ = darling.assets.energy_scan()
        reader = darling.reader.EnergyScan(
            path_to_data,
        )
        dset_energy = darling.DataSet(reader)
        mean_3d, cov_3d = dset_energy.compile_layers(
            scan_ids=["1.1", "2.1"], verbose=False
        )

        filename = os.path.join(darling.assets.path(), "saves", "energy_stack")
        dset_energy.to_paraview(filename)

    def test_visualizer_mosaicity(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)

            dset.moments()

            fig, ax = dset.plot.mosaicity(norm="full")
            fig, ax = dset.plot.mosaicity(norm="dynamic")
            fig, ax = dset.plot.mosaicity()

            if self.debug:
                fig, ax = dset.plot.mosaicity(norm="full")
                plt.show()
                fig, ax = dset.plot.mosaicity(norm="full")
                plt.show()

    def test_visualizer_covariance_with_mask(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)

            dset.moments()

            mask = np.random.choice([True, False], size=dset.data.shape[:2])
            fig, ax = dset.plot.covariance(mask=mask)

            if self.debug:
                plt.show()
    
    def check_data(self, dset):
        self.assertTrue(dset.data.dtype == np.uint16)
        self.assertTrue(len(dset.data.shape) == 4)
        self.assertTrue(dset.data.shape[2] == dset.motors.shape[1])
        self.assertTrue(dset.data.shape[3] == dset.motors.shape[2])
        self.assertTrue(dset.motors.dtype == np.float32)


if __name__ == "__main__":
    unittest.main()
    unittest.main()
    unittest.main()
