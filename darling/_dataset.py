import time

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import darling


class DataSet(object):

    def __init__(self, reader):
        self.reader = reader

    def load_scan(self, args, scan_id, roi=None):
        """Load a scan into RAM. input args should match the darling.reader.Reader used,
        however it was implemented.
        """
        self.data, self.coordinates = self.reader(args, scan_id, roi)

    def threshold(self, threshold):
        """Threshold the data based on a fixed value.

        Args:
            threshold, int : threshold value compared to individual pixels.

        """
        self.data.clip(threshold, None, out=self.data)
        self.data-=threshold

    def moments(self):
        self.mean, self.covariance = darling.properties.moments(self.data, self.coordinates)

    def get_mask(self, threshold=200):
        """Try to segment the sample based on summed intensity along motor dimensions.

        Args:
            threshold, int : value above which the sample is defined.

        Returns:
            Returns: a binary 2D maks of the sample.

        """
        mask = np.sum( self.data, axis=(2,3)) > threshold
        mask = scipy.ndimage.binary_erosion(mask, structure=np.ones((2,2)), iterations=3)
        mask = scipy.ndimage.binary_dilation(mask, structure=np.ones((2,2)), iterations=25)
        mask = scipy.ndimage.binary_fill_holes(mask)
        return mask

    def compile_layers(self, reader_args, scan_ids, threshold=None, roi=None):
        """Sequentially load a series of scans and assemble the 3D moment maps.

        this loads the mosa data array with shape N,N,m,n where N is the detector dimension and
        m,n are the motor dimensions as ordered in the self.motor_names.

        Args:
            data_name, str : path to the data without the prepended scan id
            threshold, int : background subtraction value
            scan_IDs, str : scan ids to load, e.g 1.1, 2.1 etc...
            roi, tuple of int, row_min row_max and column_min and column_max, defaults to None, in which case all data is loaded
        """
        layer_stack_mean = []
        layer_stack_covariance = []
        layer_positions = []
        for args, scan_id in zip(reader_args, scan_ids):
            print("read in scan ", scan_id)
            self.reader(args, scan_id, roi)
            if threshold:
                self.threshold(threshold)
            self.moments()
            layer_positions.append(scan_id)
            layer_stack_mean.append(self.mean)
            layer_stack_covariance.append(self.covariance)
        self.layer_stack_mean = np.array(layer_stack_mean)
        self.layer_stack_covariance = np.array(layer_stack_covariance)
        self.layer_positions = np.array(layer_positions)
        print("finished!")

if __name__ == "__main__":
    pass
