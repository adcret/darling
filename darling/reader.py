"""Collection of pre-implemneted h5 readers developed for id03 format.

NOTE: In general the file reader is strongly dependent on data collection scheme and it is therefore the
purpose of darling to allow the user to subclass Reader() and implement their own specific data structure.

Once the reader is implemented in darling format it is possible to interface the DataSet class and use
all features of darling.

"""

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np


class Reader(object):
    """Parent class for readers.

    Attributes:
        abs_path_to_h5_file (:obj: `str`): Absolute file path to data.

    """

    def __init__(self, abs_path_to_h5_file):
        self.abs_path_to_h5_file = abs_path_to_h5_file

    def read_scan(self):
        """Method to read a single 2D scan

        Returns:
            data, motors

        """
        pass


class MosaScan(Reader):
    """Load a mosa scan layer by layer.

    Args:
        abs_path_to_h5_file : str, absolute tpath to the h5 file with the diffraction images.
        motor_names : list, list of strings of paths to the data [chi, phi, strain] these need to be ordered
            to match the scan sequence.
        motor_precision : list of int, number of trusted deciamls in each motor dimension. (matching motor_names)
    """

    def __init__(
        self,
        abs_path_to_h5_file,
        motor_names,
        motor_precision,
    ):
        self.abs_path_to_h5_file = abs_path_to_h5_file
        self.motornames = motor_names
        self.motor_precision = motor_precision

        assert len(self.motor_precision) == len(
            self.motornames
        ), "The motor_precision lengths need to match the motornames length"

    def read_scan(self, data_name, scan_id, roi=None):
        """Load a scan

        this loads the mosa data array with shape N,N,m,n where N is the detector dimension and
        m,n are the motor dimensions as ordered in the self.motor_names.

        Args:
            data_name, str : path to the data wihtout the prepended scan id
            scan_id, str : scan id to load from, e.g 1.1, 2.1 etc...
            roi, tuple of int, row_min row_max and column_min and column_max, defaults to None, in which case all data is loaded
        """
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            # Read in motors
            raw_motors = [h5f[scan_id][mn] for mn in self.motornames]
            motors = [
                np.unique(np.round(m, p)).astype(np.float32)
                for p, m in zip(self.motor_precision, raw_motors)
            ]
            voxel_distribution_shape = [len(m) for m in motors]

            # read in data and reshape
            if roi:
                r1, r2, c1, c2 = roi
                data = h5f[scan_id][data_name][:, r1:r2, c1:c2]
            else:
                data = h5f[scan_id][data_name][:, :, :]

            data = self.data.reshape(
                (*voxel_distribution_shape, data.shape[-2], data.shape[-1])
            )
            data = data.swapaxes(0, 2)
            data = data.swapaxes(1, -1)
        
        return data, motors


if __name__ == "__main__":
    from darling.reader import MosaScan

    reader = MosaScan(...)
