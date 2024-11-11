import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import darling.properties
from darling.assets import mosaisicty_scan
from darling.reader import MosaScan


class TestMosaScan(unittest.TestCase):
    # Tests for the darling.MosaScan class.

    def setUp(self):
        self.debug = False
        self.path_to_data, _, _ = mosaisicty_scan()
        self.motor_names = ["instrument/diffrz/data", "instrument/chi/value"]

    def test_init(self):
        # Assert that the reader can be instantiated.
        reader = MosaScan(
            self.path_to_data,
            self.motor_names,
            motor_precision=[2, 2],
        )

    def test_read(self):
        # assert that the reader will read data and coordinates of the correct type and 
        # expected shapes.
        reader = MosaScan(
            self.path_to_data,
            self.motor_names,
            motor_precision=[2, 2],
        )

        print(reader)

        data, motors = reader.read_scan(
            data_name="instrument/pco_ff/image",
            scan_id="1.1",
        )

        self.assertTrue(data.dtype==np.uint16)
        self.assertTrue(len(data.shape)==4)
        self.assertTrue(data.shape[2]==len(motors[0]))
        self.assertTrue(data.shape[3]==len(motors[1]))
        self.assertTrue(motors[0].dtype==np.float32)
        self.assertTrue(motors[1].dtype==np.float32)
        self.assertTrue(len(motors[0].shape)==1)
        self.assertTrue(len(motors[1].shape)==1)
    
    def test_roi_read(self):
        # assert that the reader will read a roi
        reader = MosaScan(
            self.path_to_data,
            self.motor_names,
            motor_precision=[2, 2],
        )

        data, motors = reader.read_scan(
            data_name="instrument/pco_ff/image",
            scan_id="1.1",
            roi=(10, 20, 0, 7)
        )

        self.assertTrue(data.dtype==np.uint16)
        self.assertTrue(len(data.shape)==4)
        self.assertTrue(data.shape[0]==10)
        self.assertTrue(data.shape[1]==7)
        self.assertTrue(data.shape[2]==len(motors[0]))
        self.assertTrue(data.shape[3]==len(motors[1]))
        self.assertTrue(motors[0].dtype==np.float32)
        self.assertTrue(motors[1].dtype==np.float32)
        self.assertTrue(len(motors[0].shape)==1)
        self.assertTrue(len(motors[1].shape)==1)
    
    def test_scan_id(self):
        # ensure that another scan id can be read
        # assert that the reader will read a roi
        reader = MosaScan(
            self.path_to_data,
            self.motor_names,
            motor_precision=[2, 2],
        )

        data, motors = reader.read_scan(
            data_name="instrument/pco_ff/image",
            scan_id="2.1",
        )

        self.assertTrue(data.dtype==np.uint16)
        self.assertTrue(len(data.shape)==4)
        self.assertTrue(data.shape[2]==len(motors[0]))
        self.assertTrue(data.shape[3]==len(motors[1]))
        self.assertTrue(motors[0].dtype==np.float32)
        self.assertTrue(motors[1].dtype==np.float32)
        self.assertTrue(len(motors[0].shape)==1)
        self.assertTrue(len(motors[1].shape)==1)
    
if __name__ == "__main__":
    unittest.main()
