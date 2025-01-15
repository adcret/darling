import os
import unittest
import tempfile
import h5py

import matplotlib.pyplot as plt
import numpy as np

import darling

def create_mock_dataset_with_bad_motors():
    """Create a mock HDF5 dataset with intentionally bad motor values."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmpfile:
        file_path = tmpfile.name

    with h5py.File(file_path, "w") as h5f:
        # Create a scan with bad motor values
        scan_id = "1.1"
        group = h5f.create_group(scan_id)

        chi_values = np.linspace(0, 1, 100) + np.random.uniform(0, 0.0001, 100)
        phi_values = np.linspace(0, 1, 50) + np.random.uniform(0, 0.0001, 50)
        group.create_dataset("instrument/chi/value", data=chi_values)
        group.create_dataset("instrument/diffrz/data", data=phi_values)

        data_shape = (len(chi_values), len(phi_values), 100, 100)
        data = np.random.randint(0, 255, size=data_shape, dtype=np.uint16)
        group.create_dataset("instrument/pco_ff/image", data=data)

    return file_path

class TestMosaScan(unittest.TestCase):
    # Tests for the darling.MosaScan class.

    def setUp(self):
        self.debug = False
        self.path_to_data, _, _ = darling.assets.mosaicity_scan()
        self.motor_names = ["instrument/chi/value", "instrument/diffrz/data"]
        self.data_name = "instrument/pco_ff/image"

    def test_init(self):
        # Assert that the reader can be instantiated.
        reader = darling.reader.MosaScan(
            self.path_to_data,
            self.motor_names,
            motor_precision=[2, 2],
        )

    def test_read(self):
        # assert that the reader will read data and coordinates of the correct type and
        # expected shapes.
        reader = darling.reader.MosaScan(
            self.path_to_data,
            self.motor_names,
            motor_precision=[2, 2],
        )

        data, motors = reader(
            data_name=self.data_name,
            scan_id="1.1",
        )

        self.check_data(data, motors)


    def test_roi_read(self):
        # assert that the reader will read a roi
        reader = darling.reader.MosaScan(
            self.path_to_data,
            self.motor_names,
            motor_precision=[2, 2],
        )

        data, motors = reader(
            data_name=self.data_name, scan_id="1.1", roi=(10, 20, 0, 7)
        )

        self.check_data(data, motors)

        self.assertTrue(data.shape[0] == 10)
        self.assertTrue(data.shape[1] == 7)


    def test_scan_id(self):
        # ensure that another scan id can be read
        # assert that the reader will read a roi
        reader = darling.reader.MosaScan(
            self.path_to_data,
            self.motor_names,
            motor_precision=[2, 2],
        )

        data, motors = reader(
            data_name=self.data_name,
            scan_id="2.1",
        )

        data_layer_2 = data.copy()

        self.check_data(data_layer_2, motors)

        data, motors = reader(
            data_name=self.data_name,
            scan_id="1.1",
        )

        data_layer_1 = data.copy()

        # ensure the data shape is consistent between layers
        self.assertEqual(data_layer_1.shape, data_layer_2.shape)

        # ensure the data is actually different between layers.
        residual = data_layer_1 - data_layer_2
        self.assertNotEqual( np.max(np.abs(residual)), 0)


    def check_data(self, data, motors):
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(data.shape[2] == len(motors[0]))
        self.assertTrue(data.shape[3] == len(motors[1]))
        self.assertTrue(motors[0].dtype == np.float32)
        self.assertTrue(motors[1].dtype == np.float32)
        self.assertTrue(len(motors[0].shape) == 1)
        self.assertTrue(len(motors[1].shape) == 1)


class TestEnergyScan(unittest.TestCase):
    # Tests for the darling.MosaScan class.

    def setUp(self):
        self.debug = False
        self.path_to_data, _, _ = darling.assets.energy_scan()
        self.motor_names = ["instrument/positioners/ccmth", "instrument/chi/value"]
        self.motor_precision = [4, 4]
        self.data_name = "instrument/pco_ff/data"

    def test_init(self):
        # Assert that the reader can be instantiated.
        reader = darling.reader.EnergyScan(
            self.path_to_data,
            self.motor_names,
            self.motor_precision,
        )

    def test_read(self):
        # assert that the reader will read data and coordinates of the correct type and
        # expected shapes.
        reader = darling.reader.EnergyScan(
            self.path_to_data,
            self.motor_names,
            self.motor_precision,
        )

        data, motors = reader(
            data_name=self.data_name,
            scan_id="1.1",
        )

        self.check_data(data, motors)


    def test_roi_read(self):
        # assert that the reader will read a roi
        reader = darling.reader.EnergyScan(
            self.path_to_data,
            self.motor_names,
            self.motor_precision,
        )

        data, motors = reader(
            data_name=self.data_name, scan_id="1.1", roi=(10, 20, 0, 7)
        )

        self.assertTrue(data.shape[0] == 10)
        self.assertTrue(data.shape[1] == 7)
        self.check_data(data, motors)

    def test_scan_id(self):
        # ensure that another scan id can be read
        # assert that the reader will read a roi
        reader = darling.reader.EnergyScan(
            self.path_to_data,
            self.motor_names,
            self.motor_precision,
        )

        data, motors = reader(
            data_name=self.data_name,
            scan_id="2.1",
        )

        data_layer_2 = data.copy()

        self.check_data(data_layer_2, motors)

        data, motors = reader(
            data_name=self.data_name,
            scan_id="1.1",
        )

        data_layer_1 = data.copy()

        # ensure the data shape is consistent between layers
        self.assertEqual(data_layer_1.shape, data_layer_2.shape)

        # ensure the data is actually different between layers.
        residual = data_layer_1 - data_layer_2
        self.assertNotEqual( np.max(np.abs(residual)), 0)

    def check_data(self, data, motors):
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(data.shape[2] == len(motors[0]))
        self.assertTrue(data.shape[3] == len(motors[1]))
        self.assertTrue(motors[0].dtype == np.float32)
        self.assertTrue(motors[1].dtype == np.float32)
        self.assertTrue(len(motors[0].shape) == 1)
        self.assertTrue(len(motors[1].shape) == 1)

class TestWithBadMotors(unittest.TestCase):
    def setUp(self):
        self.mock_data_path = create_mock_dataset_with_bad_motors()
        self.motor_names = ["instrument/chi/value", "instrument/diffrz/data"]
        self.data_name = "instrument/pco_ff/image"
        self.scan_size = [100, 50]

    def tearDown(self):
        # Remove the mock dataset after tests
        os.remove(self.mock_data_path)

    def test_scan_size_fallback(self):
        reader = darling.reader.MosaScan(
            self.mock_data_path,
            self.motor_names,
            motor_precision=[3, 3],
        )

        data, motors = reader(
            data_name=self.data_name,
            scan_id="1.1",
            scan_size=self.scan_size,
        )


        self.assertEqual(len(motors[0]), self.scan_size[0])
        self.assertEqual(len(motors[1]), self.scan_size[1])

        self.assertAlmostEqual(motors[0][0], 0.0, places=3)
        self.assertAlmostEqual(motors[0][-1], 1.0, places=3)
        self.assertAlmostEqual(motors[1][0], 0.0, places=3)
        self.assertAlmostEqual(motors[1][-1], 1.0, places=3)

        self.assertEqual(data.shape[2], self.scan_size[0])
        self.assertEqual(data.shape[3], self.scan_size[1])

if __name__ == "__main__":
    unittest.main()
