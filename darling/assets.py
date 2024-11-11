"""Module to load example data and phantoms.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import h5py

_root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))
_asset_path = os.path.join(_root_path, "assets")


def mosaisicty_scan(scan_id='1.1'):
    """load part of a 2d mosaicity scan collected at the ESRF id03.

    This is a central detector ROI for a 111 reflection in a 5% deformed Aluminium. Two layers
    are available with scan_id 1.1 and 2.1.

    Args:
        scan_id (:obj:`str`): one of 1.1 or 2.1, specifying first or second layer scanned in the sample.

    Returns:
        data_path (:obj:`str`): absolute path to h5 file.
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) with intensity data. data[:,:,i,j] is a noisy
            detector image in type unit16 for phi and chi at index i and j respectively.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 continaning phi and chi angular cooridnates.
    """
    data_path = os.path.join(_asset_path, "example_data", "mosa_scan_id03", "mosa_scan.h5")
    with h5py.File(data_path, "r") as f:
        data = f[scan_id]["instrument/pco_ff/image"][:,:,:]
        phi = np.unique(f[scan_id]["instrument/diffrz/data"][:].round(2)).astype(np.float32)
        chi = np.unique(f[scan_id]["instrument/chi/value"][:].round(2)).astype(np.float32)
        data = data.reshape((len(phi), len(chi), data.shape[-2], data.shape[-1]))
        data = data.swapaxes(0, 2)
        data = data.swapaxes(1,-1)
    return data_path, data, (phi, chi)


def gaussian_blobs(N=32, m=9):
    """Phantom 2d scan of gaussian blobs with shifting means and covariance.

    Args:
        N,m (:obj:`int`): Desired data array size which is of shape=(m,m,N,N).

    Returns:
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) with intensity data. data[:,:,i,j] is a noisy
            detector image in type unit16 for motor x and y at index i and j respectively.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 continaning x and y cooridnates.
    """
    x = y = np.linspace(-1, 1, m, dtype=np.float32)
    sigma0 = (x[1] - x[0]) / 3.0
    X, Y = np.meshgrid(x, y, indexing="ij")
    data = np.zeros((N, N, len(x), len(y)))
    S = np.eye(2)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x0, y0 = sigma0 * i / N, sigma0 * j / N - 0.5 * sigma0 * i / N
            S[0, 0] = sigma0 + 0.5 * sigma0 * i / N
            S[1, 1] = sigma0 + 0.5 * sigma0 * j / N - 0.25 * sigma0 * i / N
            Si = 1.0 / np.diag(S)
            data[i, j] = (
                np.exp(-0.5 * (Si[0] * (X - x0) ** 2 + Si[1] * (Y - y0) ** 2)) * 64000
            )
    data = data.round().astype(np.uint16)
    return data, (x, y)


if __name__ == "__main__":
    data_path, data, coord = mosaisicty_scan()
    data, coord = gaussian_blobs(N=16, m=7)
