"""Collection of pre-implemneted h5 readers developed for id03 format.

NOTE: In general the file reader is strongly dependent on data collection scheme and it is therefore the
purpose of darling to allow the user to subclass Reader() and implement their own specific data structure.

Once the reader is implemented in darling format it is possible to interface the DataSet class and use
all features of darling.

"""

import re

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np


class Reader(object):
    """Parent class for readers.

    Args:
        abs_path_to_h5_file (:obj: `str`): Absolute file path to data.

    Attributes:
        abs_path_to_h5_file (:obj: `str`): Absolute file path to data.

    """

    def __init__(self, abs_path_to_h5_file):
        self.abs_path_to_h5_file = abs_path_to_h5_file

    def __call__(self, args, scan_id, roi=None):
        """Method to read a single 2D scan

        NOTE: This method is meant to be purpose implemented to fit the specific data aqusition
            scheme used.

        Args:
            args str (:obj:`list`): list of arguments needed by the reader.
            scan_id (:obj:`str`): scan id to load from, these are internal kayes to diffirentiate
                layers.
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded. The roi refers to the detector
                dimensions.

        Returns:
            data (:obj:`numpy array`) of shape=(a,b,m,n) and type np.uint16 and motors
            (:obj:`tuple` of :obj:`numpy array`) of shape=(m,) and shape=(n,) and type
            np.float32. a,b are detector dimensions while m,n are scan dimensions over
            which teh motor settings vary.

        """
        pass


class MosaScan(Reader):
    """Load a 2D mosa scan. This is a id03 specific implementation matching a specific beamline mosa scan macro.

    NOTE: This reader was specifically written for data collection at id03. For general purpose reading of data you
    must implement your own reader class. The exact reding of data is strongly dependent on data aqusition scheme and
    data structure implementation.

    Args:
        abs_path_to_h5_file str (:obj:`str`): absolute path to the h5 file with the diffraction images.
        motor_names data (:obj:`list` of :obj:`str`): h5 paths to the data [chi, phi, strain] these need to be ordered
            to match the scan sequence.
        motor_precision data (:obj:`list` of :obj:`int`): number of trusted deciamls in each motor dimension. (matching motor_names)
    """

    def __init__(
        self,
        abs_path_to_h5_file,
        motor_names,
        motor_precision,
    ):
        self.abs_path_to_h5_file = abs_path_to_h5_file
        self.motor_names = motor_names
        self.motor_precision = motor_precision

        assert len(self.motor_precision) == len(
            self.motor_names
        ), "The motor_precision lengths need to match the motor_names length"

    def __call__(self, data_name, scan_id, roi=None):
        """Load a scan

        this loads the mosa data array with shape N,N,m,n where N is the detector dimension and
        m,n are the motor dimensions as ordered in the self.motor_names. You may view the implemented darling readers as example templates for implementing
        your own reader.

        Args:
            data_name (:obj:`str`): path to the data wihtout the prepended scan id
            scan_id (:obj:`str`):scan id to load from, e.g 1.1, 2.1 etc...
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded

        Returns:
            data, motors : data of shape (a,b,m,n) and motors tuple of len=m and len=n

        """
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:

            # Read in motors
            raw_motors = [h5f[scan_id][mn] for mn in self.motor_names]
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

            data = data.reshape(
                (*voxel_distribution_shape, data.shape[-2], data.shape[-1])
            )
            data = data.swapaxes(0, 2)
            data = data.swapaxes(1, -1)

        return data, motors


class EnergyScan(Reader):
    """Load a 2D energy scan. This is a id03 specific implementation matching a specific beamline energy scan macro.

    NOTE: This reader was specifically written for data collection at id03. For general purpose reading of data you
    must implement your own reader class. The exact reding of data is strongly dependent on data aqusition scheme and
    data structure implementation. You may view the implemented darling readers as example templates for implementing
    your own reader.

    Args:
        abs_path_to_h5_file str (:obj:`str`): absolute path to one of the h5 file with the diffraction images. FOr an energy
            scan there is one file per layer in z. This may be any one of these paths. Provided the file name ends with
            layer_0.h5 the file path will be rebuilt upon read call.
        motor_names data (:obj:`list` of :obj:`str`): h5 paths to the data these need to be ordered exactly as [energy, chi]
            In an energy scan the h5 grouping is in energy and only the subsequent motor srquence need to be provided. I.e
            for instance 'instrument/chi/value' could be the path to the chi motor.
        motor_precision data (:obj:`list` of :obj:`int`): number of trusted deciamls in each motor dimension. (matching motor_names)
    """

    def __init__(
        self,
        abs_path_to_h5_file,
        motor_names,
        motor_precision,
    ):
        self.abs_path_to_h5_file = abs_path_to_h5_file
        self.motor_names = motor_names
        self.motor_precision = motor_precision

        assert len(self.motor_precision) == len(
            self.motor_names
        ), "The motor_precision lengths need to match the motor_names length"

    def _get_layer_path(self, scan_id):
        layer_number = str(int(scan_id[0]) - 1)
        layer_tag = r"layer_" + layer_number + ".h5"
        return re.sub(r"layer_\d+\.h5", layer_tag, self.abs_path_to_h5_file)

    def _pad_h5_paths(self, data_name, motor_names):
        mnames = []
        for name in motor_names:
            if not name.startswith("/"):
                mnames.append("/" + name)
            else:
                mnames.append(name)
        if not name.startswith("/"):
            dname = "/" + data_name
        else:
            dname = data_name
        return dname, mnames

    def __call__(self, data_name, scan_id, roi=None):
        """Load a scan

        this loads the mosa data array with shape N,N,m,n where N is the detector dimension and
        m,n are the motor dimensions as ordered in the self.motor_names.

        Args:
            data_name (:obj:`str`): path to the data wihtout the prepended energy id. i,e
                'instrument/pco_ff/data' or the like.
            scan_id (:obj:`str`):scan id to load from, e.g 1.1, 2.1 etc...
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded

        Returns:
            data, motors : data of shape (a,b,m,n) and motors tuple of len=m and len=n

        """
        abs_path_to_h5_file = self._get_layer_path(scan_id)
        dname, mnames = self._pad_h5_paths(data_name, self.motor_names)

        with h5py.File(abs_path_to_h5_file, "r") as h5f:
            key0 = list(h5f.keys())[0]

            chi = h5f[key0 + mnames[1]][:].astype(np.float32)

            _, det_rows, det_cols = h5f[key0 + dname].shape
            n_energy = len(h5f.keys())
            n_chis = len(chi)

            if roi is None:
                data = np.zeros((det_rows, det_cols, n_energy, n_chis), dtype=np.uint16)
            else:
                r1, r2, c1, c2 = roi
                data = np.zeros((r2 - r1, c2 - c1, n_energy, n_chis), dtype=np.uint16)

            energy = np.zeros((n_energy,), dtype=np.float32)

            all_chis = []

            for i, key in enumerate(h5f.keys()):  # iterates over energies.
                chi_stack = h5f[key + dname][:, :, :]
                chi_stack = np.swapaxes(chi_stack, 0, 1)
                chi_stack = np.swapaxes(chi_stack, 1, 2)
                if roi is None:
                    data[:, :, i, :] = chi_stack
                else:
                    data[:, :, i, :] = chi_stack[r1:r2, c1:c2, :]
                energy[i] = h5f[key + mnames[0]][()]
                all_chis.extend(list(h5f[key + mnames[1]][:]))

        chi = np.unique(np.round(all_chis, self.motor_precision[0])).astype(np.float32)
        energy = np.round(energy, self.motor_precision[0]).astype(np.float32)
        motors = [energy, chi]

        assert len(chi) == data.shape[3], "Potential motor drift in chi"

        return data, motors


if __name__ == "__main__":
    pass
