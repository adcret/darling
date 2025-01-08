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
        self.reader_1 = darling.reader.MosaScan(
            path_to_data_1,
            ["instrument/diffrz/data", "instrument/chi/value"],
            motor_precision=[2, 2],
        )
        self.data_name_1 = "instrument/pco_ff/image"

        # as well as the energy scan reader
        path_to_data_2, _, _ = darling.assets.energy_scan()
        self.reader_2 = darling.reader.EnergyScan(
            path_to_data_2,
            ["/instrument/positioners/ccmth", "/instrument/chi/value"],
            [4, 4],
        )
        self.data_name_2 =  "/instrument/pco_ff/data"

        self.readers = [self.reader_1, self.reader_2]
        self.data_names = [self.data_name_1, self.data_name_2]

    def test_init(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)

    def test_load_scan(self):
        for reader, data_name in zip(self.readers, self.data_names):
            dset = darling.DataSet(reader)

            # test that a scan can be loaded.
            args = data_name
            dset.load_scan(args, scan_id="1.1", roi=None)
            self.check_data(dset)

            # test the tuple args option
            args = (data_name,)
            dset.load_scan(args, scan_id="1.1", roi=None)
            self.check_data(dset)
            data_layer_1 = dset.data.copy()

            # test to load a diffrent layer
            args = (data_name,)
            dset.load_scan(args, scan_id="2.1", roi=None)
            self.check_data(dset)

            # ensure the data shape is consistent between layers
            self.assertEqual(data_layer_1.shape, dset.data.shape)

            # ensure the data is actually different between layers.
            residual = data_layer_1 - dset.data
            self.assertNotEqual( np.max(np.abs(residual)), 0)

            # test that a roi can be loaded and that the resulting shape is ok.
            args = (data_name,)
            dset.load_scan(args, scan_id="2.1", roi=(0, 9, 3, 19))
            self.check_data(dset)
            self.assertTrue(dset.data.shape[0] == 9)
            self.assertTrue(dset.data.shape[1] == 16)

    def test_subtract(self):
        for reader, data_name in zip(self.readers, self.data_names):
            dset = darling.DataSet(reader)
            dset.load_scan(data_name, scan_id="1.1", roi=None)
            mm = np.max(dset.data)
            dset.subtract(value=200)
            self.assertEqual(np.max(dset.data), mm - 200)

    def test_moments(self):
        for reader, data_name in zip(self.readers, self.data_names):
            dset = darling.DataSet(reader)
            dset.load_scan(data_name, scan_id="1.1", roi=None)
            mean, covariance = dset.moments()
            self.assertEqual(mean.shape[0], dset.data.shape[0])
            self.assertEqual(mean.shape[1], dset.data.shape[1])
            self.assertEqual(mean.shape[2], 2)
            self.assertEqual(covariance.shape[0], dset.data.shape[0])
            self.assertEqual(covariance.shape[1], dset.data.shape[1])
            self.assertEqual(covariance.shape[2], 2)
            self.assertEqual(covariance.shape[3], 2)

            if self.debug:
                dset.plot.mean()

    def test_estimate_mask(self):
        for reader, data_name in zip(self.readers, self.data_names):
            dset = darling.DataSet(reader)
            dset.load_scan(data_name, scan_id="1.1", roi=None)
            mask = dset.estimate_mask()
            self.assertEqual(mask.shape[0], dset.data.shape[0])
            self.assertEqual(mask.shape[1], dset.data.shape[1])
            self.assertEqual(mask.dtype, bool)

            if self.debug:
                plt.style.use('dark_background')
                fig, ax = plt.subplots(1, 1, figsize=(7,7))
                im = ax.imshow(mask)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show()

    def test_integrate(self):
        for reader, data_name in zip(self.readers, self.data_names):
            dset = darling.DataSet(reader)
            dset.load_scan(data_name, scan_id="1.1", roi=None)
            int_frames = dset.integrate()
            self.assertEqual(int_frames.shape[0], dset.data.shape[0])
            self.assertEqual(int_frames.shape[1], dset.data.shape[1])
            self.assertEqual(int_frames.dtype, np.float32)

            if self.debug:
                plt.style.use('dark_background')
                fig, ax = plt.subplots(1, 1, figsize=(7,7))
                im = ax.imshow(int_frames)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show()
    
    def test_compile_layers(self):

        # test that the mosa reader will stack layers
        path_to_data, _, _ = darling.assets.mosaicity_scan()
        reader = darling.reader.MosaScan(
            path_to_data,
            ["instrument/chi/value", "instrument/diffrz/data"],
            motor_precision=[3, 3],
        )

        data_name = "instrument/pco_ff/image"
        dset_mosa = darling.DataSet(reader)

        mean_3d, cov_3d = dset_mosa.compile_layers(
            data_name, scan_ids=["1.1", "2.1"], verbose=False
        )

        self.assertEqual(mean_3d.shape[0], 2)
        self.assertEqual(len(mean_3d.shape), 4)
        self.assertEqual(cov_3d.shape[0], 2)
        self.assertEqual(len(cov_3d.shape), 5)

        # test that the energy reader can stack layers.
        path_to_data, _, _ = darling.assets.energy_scan()
        data_name = "instrument/pco_ff/data"
        reader = darling.reader.EnergyScan(
        path_to_data,
        ["instrument/positioners/ccmth", "instrument/chi/value"],
        motor_precision=[4, 4],
        )
        dset_energy = darling.DataSet(reader)
        mean_3d, cov_3d = dset_energy.compile_layers(
            data_name, scan_ids=["1.1", "2.1"], verbose=False
        )

        self.assertEqual(mean_3d.shape[0], 2)
        self.assertEqual(len(mean_3d.shape), 4)
        self.assertEqual(cov_3d.shape[0], 2)
        self.assertEqual(len(cov_3d.shape), 5)

    def test_as_paraview(self):
        path_to_data, _, _ = darling.assets.mosaicity_scan()
        reader = darling.reader.MosaScan(
            path_to_data,
            ["instrument/chi/value", "instrument/diffrz/data"],
            motor_precision=[3, 3],
        )

        data_name = "instrument/pco_ff/image"
        dset_mosa = darling.DataSet(reader)

        mean_3d, cov_3d = dset_mosa.compile_layers(
            data_name, scan_ids=["1.1", "2.1"], verbose=False
        )

        filename = os.path.join(darling.assets.path(), 'saves', 'mosa_stack')
        dset_mosa.to_paraview(filename)

        path_to_data, _, _ = darling.assets.energy_scan()
        data_name = "instrument/pco_ff/data"
        reader = darling.reader.EnergyScan(
        path_to_data,
        ["instrument/positioners/ccmth", "instrument/chi/value"],
        motor_precision=[4, 4],
        )
        dset_energy = darling.DataSet(reader)
        mean_3d, cov_3d = dset_energy.compile_layers(
            data_name, scan_ids=["1.1", "2.1"], verbose=False
        )

        filename = os.path.join(darling.assets.path(), 'saves', 'energy_stack')
        dset_energy.to_paraview(filename)


    def check_data(self, dset):
        self.assertTrue(dset.data.dtype == np.uint16)
        self.assertTrue(len(dset.data.shape) == 4)
        self.assertTrue(dset.data.shape[2] == len(dset.motors[0]))
        self.assertTrue(dset.data.shape[3] == len(dset.motors[1]))
        self.assertTrue(dset.motors[0].dtype == np.float32)
        self.assertTrue(dset.motors[1].dtype == np.float32)
        self.assertTrue(len(dset.motors[0].shape) == 1)
        self.assertTrue(len(dset.motors[1].shape) == 1)

    def test_visualizer_mosaicity(self):
        for reader, data_name in zip(self.readers, self.data_names):
            dset = darling.DataSet(reader)
            dset.load_scan(data_name, scan_id="1.1", roi=None)
            
            dset.moments()

            dset.plot.mosaicity(use_motors=False)

            dset.plot.mosaicity(use_motors=True)

            mask = np.random.choice([True, False], size=dset.data.shape[:2])
            dset.plot.mosaicity(use_motors=True, mask=mask)

    def test_visualizer_covariance_with_mask(self):
        for reader, data_name in zip(self.readers, self.data_names):
            dset = darling.DataSet(reader)
            dset.load_scan(data_name, scan_id="1.1", roi=None)

            dset.moments()

            mask = np.random.choice([True, False], size=dset.data.shape[:2])
            dset.plot.covariance(mask=mask)
if __name__ == "__main__":
    unittest.main()
