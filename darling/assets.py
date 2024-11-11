"""Module to load example data and phantoms.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

_root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))
_asset_path = os.path.join(_root_path, "assets")

def mosaisicty_scan():
    """load part of a 2d mosaicity scan collected at the ESRF id03.

    Returns:
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) with intensity data. data[:,:,i,j] is a noisy 
            detector image in type unit16 for phi and chi at index i and j respectively.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 continaning phi and chi angular cooridnates.
    """
    data_path = os.path.join(_asset_path, "example_data", "mosa_scan_id03")
    data = np.load(os.path.join(data_path, "sample_data.npy"))
    phi = np.load(os.path.join(data_path, "phi.npy"))
    chi = np.load(os.path.join(data_path, "chi.npy"))
    return data, (phi, chi)

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
                np.exp(-0.5 * (Si[0] * (X - x0) ** 2 + Si[1] * (Y - y0) ** 2))
                * 64000
            )
    data = data.round().astype(np.uint16)
    return data, (x, y)


if __name__ == "__main__":
    data, coord = mosaisicty_scan()
    data, coord = gaussian_blobs(N=16, m=7)
