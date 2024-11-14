"""Functions module for computation of data features over 4D or 5D fields. I.e computation of moments of 
mosa-scans strain-mosa-scans and the like.

As an example, in a DFXM strain-mosaicity-scan setting, using random arrays, the 3D moments 
in theta, phi and chi can be retrieved as:

.. code-block:: python

    import numpy as np
    import darling

    # create coordinate arrays
    theta = np.linspace(-1, 1, 9) # crl scan grid
    phi = np.linspace(-1, 1, 8) # motor rocking scan grid
    chi = np.linspace(-1, 1, 16) # motor rolling scan grid

    # create a random data array
    detector_dim = (128, 128) # the number of rows and columns of the detector
    data = np.random.rand(*detector_dim, len(phi), len(chi), len(theta))

    data = data.astype(np.uint16) # the collected intensity data for the entire scan

    # compute the first and second moments such that
    # mean[i,j] is the shape=(3,) array of mean coorindates for pixel i,j.
    # covariance[i,j] is the shape=(3,3) covariance matrix of pixel i,j.
    mean, covariance = darling.properties.moments(data, coordinates=(phi, chi, theta))

    assert mean.shape==(128, 128, 3)
    assert covariance.shape==(128, 128, 3, 3)
"""

import numba
import numpy as np

def moments(data, coordinates):
    """Compute the sample mean and covariance of a 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 2 or 3 degrees of freedom. These could be phi and chi or phi and energy, etc.
    The total data array is therefore either 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

    Example in a DFXM mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)

        # create a random data array
        detector_dim = (128, 128)
        data = np.random.rand(*detector_dim, len(phi), len(chi))

        data = data.astype(np.uint16)

        # compute the first and second moments
        mean, covariance = darling.properties.moments(data, coordinates=(phi, chi))

    Args:
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) or shape=(n, m, o, a, b) where the maps over which the mean will 
            be calculated are of shape=(n, m) or shape=(m, n, o) and the field is of shape=(a, b). Must be numpy uint16. I.e the
            detector roi is of shape=(a,b) while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution
            for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 or len==3 continaning numpy 1d arrays specifying the 
            coordinates in the n and m dimensions respectively. I.e, as an example, as an example these could be the phi and chi
            angular cooridnates.

    Returns:
        :obj:`tuple` of :obj:`numpy array` : The mean map of shape=(k,l) and the covariance map of shape=(k,l,2,2).
    """
    mu = mean(data, coordinates)
    cov = covariance(data, coordinates, first_moments=mu)
    return mu, cov


def mean(data, coordinates):
    """Compute the sample mean of a 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 2 or 3 degrees of freedom. These could be phi and chi or phi and energy, etc.
    The total data array is therefore either 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

    Example in a DFXM mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        theta = np.linspace(-1, 1, 7)
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)

        # create a random data array
        detector_dim = (128, 128)
        data = np.random.rand(*detector_dim, len(theta), len(phi), len(chi))

        data = data.astype(np.uint16)

        # compute the first moments
        first_moment = darling.properties.mean(data, coordinates=(theta, phi, chi))

    Args:
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) or shape=(n, m, o, a, b) where the maps over which the mean will be calculated are
            of shape=(n, m) or shape=(m, n, o) and the field is of shape=(a, b). Must be numpy uint16. I.e the detector roi is of shape=(a,b)
            while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 or len=3 continaning numpy 1d arrays specifying the coordinates
            in the n and m dimensions respectively. I.e, as an example, these could be the
            phi and chi angular cooridnates.

    Returns:
        :obj:`numpy array` : The mean map of shape=(k,l).
    """
    _check_data(data, coordinates)
    dum_1 = np.arange(1).astype(np.float32)
    dum_2 = np.arange(len(coordinates)).astype(np.float32)
    if len(coordinates) == 2:
        x, y = coordinates
        a, b, m, n = data.shape
        xr = x.reshape(m, 1).astype(np.float32)
        yr = y.reshape(1, n).astype(np.float32)
        res = np.zeros((a, b, 2), dtype=np.float32)
        _first_moments2D(data, xr, yr, dum_1, dum_2, res)
    else:
        x, y, z = coordinates
        a, b, m, n, o = data.shape
        xr = x.reshape(m, 1, 1).astype(np.float32)
        yr = y.reshape(1, n, 1).astype(np.float32)
        zr = z.reshape(1, 1, o).astype(np.float32)
        res = np.zeros((a, b, 3), dtype=np.float32)
        _first_moments3D(data, xr, yr, zr, dum_1, dum_2, res)
    return res


def covariance(data, coordinates, first_moments=None):
    """Compute the sample covariance of a 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 2 or 3 degrees of freedom. These could be phi and chi or phi and energy, etc.
    The total data array is therefore either 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

    Example in a DFXM mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)

        # create a random data array
        detector_dim = (128, 128)
        data = np.random.rand(*detector_dim, len(phi), len(chi))

        data = data.astype(np.uint16)

        # compute the first moments
        first_moment = darling.properties.mean(data, coordinates=(phi, chi))

        # compute the second moments
        covariance = darling.properties.covariance(data, (phi, chi), first_moments=first_moment)

    Args:
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) or shape=(n, m, o, a, b) where the maps over which the covariance will
            be calculated are of shape=(n, m) or shape=(n, m, o) and the field is of shape=(a, b). Must be numpy uint16. I.e the detector
            roi is of shape=(a,b) while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 or len=3 continaning numpy 1d arrays specifying the coordinates
            in the n and m (and o) dimensions respectively. I.e, as an example, these could be the phi and chi angular cooridnates.
        first_moments (:obj:`numpy array`):  Array of shape=(a, b) of the first moments. Defaults to None, in which case the first moments
            are recomputed on the fly.

    Returns:
        :obj:`numpy array` : The covariance map of shape=(k,l,2,2).
    """
    _check_data(data, coordinates)
    dim = len(coordinates)
    dum = np.arange(dim).astype(np.float32)
    res = np.zeros((data.shape[0], data.shape[1], dim, dim), dtype=np.float32)
    points = _get_point_mesh(coordinates)
    if first_moments is None:
        first_moments = mean(data, coordinates)
    if dim==2:
        _second_moments2D(data, first_moments, points, dum, res)
    elif dim==3:
        _second_moments3D(data, first_moments, points, dum, res)
    return res

def _check_data(data, coordinates):
    assert data.dtype == np.uint16, "data must be of type uint16"
    if len(coordinates) == 2:
        assert len(data.shape) == 4, "2D scan data array must be of shape=(a, b, n, m)"
    elif len(coordinates) == 3:
        assert (
            len(data.shape) == 5
        ), "3D scan data array must be of shape=(a, b, n, m, o)"
    else:
        raise ValueError("Coordinate array must be 2D or 3D")

def _get_point_mesh(coordinates):
    mesh = np.meshgrid(*coordinates, indexing="ij")
    points = np.zeros((len(coordinates), mesh[0].size))
    for i in range(len(coordinates)):
        points[i,:] = mesh[i].flatten()
    return points.astype(np.float32)

@numba.guvectorize(
    [
        (
            numba.uint16[:, :],
            numba.float32[:, :],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:],
            numba.float32[:],
        )
    ],
    "(m,n),(m,l),(l,n),(l),(p)->(p)",
    nopython=True,
    target="parallel",
)
def _first_moments2D(data, x, y, dum_1, dum_2, res):
    """Compute the sample mean of a 2D map.

    Args:
        data (:obj:`numpy array`): a 2d data map to proccess.
        chis (:obj:`numpy array`): the first coordinate array
        phis (:obj:`numpy array`): the second coordinate array
        dum (:obj:`numpy array`): dummpy variable for numba shapes. (len=2)
        res (:obj:`numpy array`): array in which to store output.
    """
    I = np.sum(data)
    if I == 0:
        res[...] = np.zeros((2,))
    else:
        com_x = np.sum(data * x) / I
        com_y = np.sum(data * y) / I
        res[...] = [com_x, com_y]


@numba.guvectorize(
    [
        (
            numba.uint16[:, :, :],
            numba.float32[:, :, :],
            numba.float32[:, :, :],
            numba.float32[:, :, :],
            numba.float32[:],
            numba.float32[:],
            numba.float32[:],
        )
    ],
    "(m,n,o),(m,l,l),(l,n,l),(l,l,o),(l),(p)->(p)",
    nopython=True,
    target="parallel",
)
def _first_moments3D(data, x, y, z, dum_1, dum_2, res):
    """Compute the sample mean of a 3D map.

    Args:
        data (:obj:`numpy array`): a 3d data map to proccess.
        x (:obj:`numpy array`): the first coordinate array.
        y (:obj:`numpy array`): the second coordinate array.
        z (:obj:`numpy array`): the third coordinate array.
        dum_1 (:obj:`numpy array`): dummy variable for numba shapes. (of shape 1)
        dum_2 (:obj:`numpy array`): dummy variable for numba shapes. (of shape 3)
        res (:obj:`numpy array`): array in which to store output.
    """
    I = np.sum(data)
    if I == 0:
        res[...] = np.zeros((3,))
    else:
        com_x = np.sum(data * x) / I
        com_y = np.sum(data * y) / I
        com_z = np.sum(data * z) / I
        res[...] = [com_x, com_y, com_z]


@numba.guvectorize(
    [
        (
            numba.uint16[:, :],
            numba.float32[:],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:, :],
        )
    ],
    "(n,m),(p),(k,q),(p)->(p,p)",
    nopython=True,
    target="parallel",
)
def _second_moments2D(chi_phi, first_moments, points, dum, res):
    """Compute the sample covariance of a 2D map.

    Args:
        chi_phi (:obj:`numpy array`): a 2d data map to proccess.
        first_moments (:obj:`numpy array`): the first moment of the 2d data map to proccess.
        points (:obj:`numpy array`): 2,n flattened array of the coordinates.
        dum (:obj:`numpy array`): dummpy variable for numba shapes.
        res (:obj:`numpy array`): array in which to store output.
    """
    I = np.sum(chi_phi)
    if I == 0:
        res[...] = np.zeros((2, 2))
    else:
        # Equivalent of the numpy.cov function setting the chi_phi intesnity as aweights,
        # see also https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        m = points.copy()
        m[0] -= first_moments[0]
        m[1] -= first_moments[1]
        a = chi_phi.flatten()
        m -= np.sum(m * a) / I
        cov = np.dot(m * a, m.T) / I
        res[...] = cov


@numba.guvectorize(
    [
        (
            numba.uint16[:, :, :],
            numba.float32[:],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:, :],
        )
    ],
    "(n,m,o),(p),(k,q),(p)->(p,p)",
    nopython=True,
    target="parallel",
)
def _second_moments3D(data, first_moments, points, dum, res):
    """Compute the sample covariance of a 3D map.

    Args:
        data (:obj:`numpy array`): a 3d data map to proccess.
        first_moments (:obj:`numpy array`): the first moment of the 2d data map to proccess.
        points (:obj:`numpy array`): 2,n flattened array of the coordinates.
        dum (:obj:`numpy array`): dummy variable for numba shapes. (of shape 3)
        res (:obj:`numpy array`): array in which to store output.
    """
    I = np.sum(data)
    if I == 0:
        res[...] = np.zeros((3, 3))
    else:
        # Equivalent of the numpy.cov function setting the chi_phi intesnity as aweights,
        # see also https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        m = points.copy()
        m[0] -= first_moments[0]
        m[1] -= first_moments[1]
        m[2] -= first_moments[2]
        a = data.flatten()
        m -= np.sum(m * a) / I
        cov = np.dot(m * a, m.T) / I
        res[...] = cov

if __name__ == "__main__":
    pass