"""Parametric fitting utilities for Gaussian intensity distributions."""

import numba
import numpy as np


def gaussian(data, coordinates, mode="full", k=3, labels=None, filter=None):
    """Fit one full-covariance Gaussian to every detector pixel.

    The fitted model is

    ``I(x) = amplitude * exp(-0.5 * (x - mean).T @ precision @ (x - mean))``

    where ``x`` and ``mean`` are vectors of length 2 or 3. The same linear
    algebraic definition is used for both dimensions; the public wrapper only
    dispatches to storage-specific numba kernels.

    When ``mode="full"`` the fit uses every positive intensity value in each
    detector pixel spectrum. When ``mode="local_max"`` each spectrum is first
    segmented with :func:`darling.properties.local_max_label`, and the ``k``
    strongest labelled regions are fit independently. Precomputed ``labels``
    can also be supplied to fit labelled regions directly.

    Args:
        data (:obj:`numpy.ndarray`): Intensity array of shape ``(a, b, m, n)``
            or ``(a, b, m, n, o)``. Must be ``uint16``. The final 2 or 3 axes
            are the coordinates over which a Gaussian is fit for each detector
            pixel ``data[i, j, ...]``.
        coordinates (:obj:`tuple` or :obj:`numpy.ndarray`): Coordinate grids
            with length 2 or 3. Each coordinate array must match
            ``data.shape[2:]``.
        mode (:obj:`str`): Either ``"full"`` or ``"local_max"``.
            Defaults to ``"full"``.
        k (:obj:`int`): Number of labelled regions to fit per detector pixel
            when ``mode="local_max"`` or ``labels`` are supplied. Defaults to 3.
        labels (:obj:`numpy.ndarray`): Optional label array with the same shape
            as ``data``. Labels must be positive integers, with zero denoting
            background/void.
        filter (:obj:`dict`): Optional local-max labelling filter dictionary,
            forwarded to :func:`darling.properties.local_max_label` when
            ``mode="local_max"``.

    Returns:
        :obj:`dict`: Fitted Gaussian parameters with keys:

        - ``amplitude``: shape ``(a, b)``
        - ``mean``: shape ``(a, b, dim)``
        - ``covariance``: shape ``(a, b, dim, dim)``
        - ``precision``: shape ``(a, b, dim, dim)``
        - ``log_residual``: mean squared residual in log-intensity space,
          shape ``(a, b)``

        In labelled mode, these arrays include a peak axis after the detector
        axes: ``amplitude`` has shape ``(a, b, k)``, ``mean`` has shape
        ``(a, b, k, dim)`` and covariance/precision have shape
        ``(a, b, k, dim, dim)``. Labelled mode also returns ``label``,
        ``sum_intensity`` and ``number_of_pixels``.

        Fits with too few positive samples or non-positive-definite fitted
        precision are filled with zeros.
    """
    _check_gaussian_data(data, coordinates)
    coordinates = tuple(coordinates)
    dim = len(coordinates)

    if labels is not None or mode in ("local_max", "labels", "labeled"):
        if k <= 0:
            raise ValueError("k must be larger than 0")
        if labels is None:
            from darling.properties import local_max_label

            labels, _ = local_max_label(data, loop_outer_dims=True, filter=filter)
        _check_labels(data, labels)
        return _labeled_gaussian(data, coordinates, labels, k)

    if mode != "full":
        raise ValueError('mode must be "full" or "local_max"')

    amplitude = np.zeros(data.shape[:2], dtype=np.float32)
    mean = np.zeros((*data.shape[:2], dim), dtype=np.float32)
    covariance = np.zeros((*data.shape[:2], dim, dim), dtype=np.float32)
    precision = np.zeros((*data.shape[:2], dim, dim), dtype=np.float32)
    log_residual = np.zeros(data.shape[:2], dtype=np.float32)

    if dim == 2:
        _fit_gaussian2d(
            data,
            coordinates[0],
            coordinates[1],
            amplitude,
            mean,
            covariance,
            precision,
            log_residual,
        )
    elif dim == 3:
        _fit_gaussian3d(
            data,
            coordinates[0],
            coordinates[1],
            coordinates[2],
            amplitude,
            mean,
            covariance,
            precision,
            log_residual,
        )
    else:
        raise ValueError("Gaussian fitting is implemented for 2D and 3D coordinates")

    return {
        "amplitude": amplitude,
        "mean": mean,
        "covariance": covariance,
        "precision": precision,
        "log_residual": log_residual,
    }


def _check_gaussian_data(data, coordinates):
    if data.dtype != np.uint16:
        raise AssertionError("data must be of type uint16")
    if len(coordinates) not in (2, 3):
        raise ValueError("coordinates must contain 2 or 3 coordinate arrays")
    if data.ndim != len(coordinates) + 2:
        raise AssertionError("data shape must be (a, b, ...) with 2 or 3 scan axes")
    for c in coordinates:
        if not isinstance(c, np.ndarray):
            raise ValueError("Coordinate array must be a numpy array")
        if c.shape != data.shape[2:]:
            raise AssertionError("coordinate array do not match data shape")


def _check_labels(data, labels):
    if not isinstance(labels, np.ndarray):
        raise ValueError("labels must be a numpy array")
    if labels.shape != data.shape:
        raise AssertionError("labels must have the same shape as data")
    if labels.dtype != np.uint16:
        raise AssertionError("labels must be of type uint16")


def _labeled_gaussian(data, coordinates, labels, k):
    dim = len(coordinates)
    max_labels = int(np.max(labels))
    amplitude = np.zeros((*data.shape[:2], k), dtype=np.float32)
    mean = np.zeros((*data.shape[:2], k, dim), dtype=np.float32)
    covariance = np.zeros((*data.shape[:2], k, dim, dim), dtype=np.float32)
    precision = np.zeros((*data.shape[:2], k, dim, dim), dtype=np.float32)
    log_residual = np.zeros((*data.shape[:2], k), dtype=np.float32)
    label = np.zeros((*data.shape[:2], k), dtype=np.uint16)
    sum_intensity = np.zeros((*data.shape[:2], k), dtype=np.float32)
    number_of_pixels = np.zeros((*data.shape[:2], k), dtype=np.uint16)

    if dim == 2:
        _fit_labeled_gaussian2d(
            data,
            coordinates[0],
            coordinates[1],
            labels,
            max_labels,
            amplitude,
            mean,
            covariance,
            precision,
            log_residual,
            label,
            sum_intensity,
            number_of_pixels,
        )
    elif dim == 3:
        _fit_labeled_gaussian3d(
            data,
            coordinates[0],
            coordinates[1],
            coordinates[2],
            labels,
            max_labels,
            amplitude,
            mean,
            covariance,
            precision,
            log_residual,
            label,
            sum_intensity,
            number_of_pixels,
        )
    else:
        raise ValueError("Gaussian fitting is implemented for 2D and 3D coordinates")

    return {
        "amplitude": amplitude,
        "mean": mean,
        "covariance": covariance,
        "precision": precision,
        "log_residual": log_residual,
        "label": label,
        "sum_intensity": sum_intensity,
        "number_of_pixels": number_of_pixels,
    }


@numba.njit(cache=True)
def _solve_linear_system(a, b, n, x):
    """Solve ``a @ x = b`` in place using Gaussian elimination."""
    for k in range(n):
        pivot = k
        pivot_abs = abs(a[k, k])
        for i in range(k + 1, n):
            candidate = abs(a[i, k])
            if candidate > pivot_abs:
                pivot = i
                pivot_abs = candidate

        if pivot_abs < 1e-12:
            return False

        if pivot != k:
            tmp = b[k]
            b[k] = b[pivot]
            b[pivot] = tmp
            for j in range(k, n):
                tmp = a[k, j]
                a[k, j] = a[pivot, j]
                a[pivot, j] = tmp

        for i in range(k + 1, n):
            factor = a[i, k] / a[k, k]
            a[i, k] = 0.0
            for j in range(k + 1, n):
                a[i, j] -= factor * a[k, j]
            b[i] -= factor * b[k]

    for i in range(n - 1, -1, -1):
        value = b[i]
        for j in range(i + 1, n):
            value -= a[i, j] * x[j]
        if abs(a[i, i]) < 1e-12:
            return False
        x[i] = value / a[i, i]

    return True


@numba.njit(cache=True)
def _copy_square(src, dst, n):
    for i in range(n):
        for j in range(n):
            dst[i, j] = src[i, j]


@numba.njit(cache=True)
def _fit_quadratic_to_gaussian(dim, params, amplitude, mean, covariance, precision):
    """Convert fitted log-quadratic parameters to Gaussian parameters."""
    p = np.zeros((3, 3), dtype=np.float64)
    p[0, 0] = -2.0 * params[dim + 1]
    p[0, 1] = -params[dim + 2]
    p[1, 0] = p[0, 1]

    if dim == 2:
        p[1, 1] = -2.0 * params[dim + 3]
    else:
        p[0, 2] = -params[dim + 3]
        p[2, 0] = p[0, 2]
        p[1, 1] = -2.0 * params[dim + 4]
        p[1, 2] = -params[dim + 5]
        p[2, 1] = p[1, 2]
        p[2, 2] = -2.0 * params[dim + 6]

    for i in range(dim):
        if p[i, i] <= 0.0:
            return False

    system = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    solution = np.zeros(3, dtype=np.float64)
    _copy_square(p, system, dim)
    for i in range(dim):
        rhs[i] = params[i + 1]

    if not _solve_linear_system(system, rhs, dim, solution):
        return False

    for i in range(dim):
        mean[i] = solution[i]
        for j in range(dim):
            precision[i, j] = p[i, j]

    quad = 0.0
    for i in range(dim):
        for j in range(dim):
            quad += solution[i] * p[i, j] * solution[j]
    amplitude[0] = np.exp(params[0] + 0.5 * quad)

    for col in range(dim):
        _copy_square(p, system, dim)
        for i in range(dim):
            rhs[i] = 0.0
            solution[i] = 0.0
        rhs[col] = 1.0
        if not _solve_linear_system(system, rhs, dim, solution):
            return False
        for row in range(dim):
            covariance[row, col] = solution[row]

    for i in range(dim):
        if covariance[i, i] <= 0.0:
            return False

    return True


@numba.njit(cache=True)
def _reset_fit(dim, amplitude, mean, covariance, precision, log_residual):
    amplitude[0] = 0.0
    log_residual[0] = 0.0
    for i in range(dim):
        mean[i] = 0.0
        for j in range(dim):
            covariance[i, j] = 0.0
            precision[i, j] = 0.0


@numba.njit(cache=True)
def _accumulate_normal_equations(phi, q, log_intensity, normal, rhs):
    for r in range(q):
        rhs[r] += phi[r] * log_intensity
        for s in range(q):
            normal[r, s] += phi[r] * phi[s]


@numba.njit(cache=True)
def _finish_fit(dim, q, count, normal, rhs, amplitude, mean, covariance, precision):
    if count < q:
        return False

    # Small diagonal damping stabilizes nearly singular coordinate grids without
    # changing well-conditioned fits at float32 precision.
    for r in range(q):
        normal[r, r] += 1e-10

    params = np.zeros(10, dtype=np.float64)
    if not _solve_linear_system(normal, rhs, q, params):
        return False

    return _fit_quadratic_to_gaussian(
        dim, params, amplitude, mean, covariance, precision
    )


@numba.njit(cache=True)
def _gaussian_log_value(dim, x, amplitude, mean, precision):
    if amplitude[0] <= 0.0:
        return 0.0

    quad = 0.0
    for i in range(dim):
        dxi = x[i] - mean[i]
        for j in range(dim):
            quad += dxi * precision[i, j] * (x[j] - mean[j])
    return np.log(amplitude[0]) - 0.5 * quad


@numba.njit(cache=True, parallel=True)
def _fit_gaussian2d(data, x, y, amplitude, mean, covariance, precision, log_residual):
    dim = 2
    q = 6
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            normal = np.zeros((10, 10), dtype=np.float64)
            rhs = np.zeros(10, dtype=np.float64)
            phi = np.zeros(10, dtype=np.float64)
            count = 0

            for ii in range(data.shape[2]):
                for jj in range(data.shape[3]):
                    value = data[i, j, ii, jj]
                    if value == 0:
                        continue

                    x0 = x[ii, jj]
                    x1 = y[ii, jj]
                    phi[0] = 1.0
                    phi[1] = x0
                    phi[2] = x1
                    phi[3] = x0 * x0
                    phi[4] = x0 * x1
                    phi[5] = x1 * x1
                    _accumulate_normal_equations(
                        phi, q, np.log(float(value)), normal, rhs
                    )
                    count += 1

            if not _finish_fit(
                dim,
                q,
                count,
                normal,
                rhs,
                amplitude[i, j : j + 1],
                mean[i, j],
                covariance[i, j],
                precision[i, j],
            ):
                _reset_fit(
                    dim,
                    amplitude[i, j : j + 1],
                    mean[i, j],
                    covariance[i, j],
                    precision[i, j],
                    log_residual[i, j : j + 1],
                )
                continue

            residual = 0.0
            for ii in range(data.shape[2]):
                for jj in range(data.shape[3]):
                    value = data[i, j, ii, jj]
                    if value == 0:
                        continue
                    phi[0] = x[ii, jj]
                    phi[1] = y[ii, jj]
                    diff = np.log(float(value)) - _gaussian_log_value(
                        dim,
                        phi,
                        amplitude[i, j : j + 1],
                        mean[i, j],
                        precision[i, j],
                    )
                    residual += diff * diff
            log_residual[i, j] = residual / count


@numba.njit(cache=True)
def _insert_label_by_intensity(label_value, intensity, labels, intensities):
    for pos in range(labels.shape[0]):
        if intensity > intensities[pos]:
            for shift in range(labels.shape[0] - 1, pos, -1):
                labels[shift] = labels[shift - 1]
                intensities[shift] = intensities[shift - 1]
            labels[pos] = label_value
            intensities[pos] = intensity
            return


@numba.njit(cache=True)
def _fit_labeled_region(dim, q, count, normal, rhs, amp, mu, cov, prec):
    if count < q:
        return False
    return _finish_fit(dim, q, count, normal, rhs, amp, mu, cov, prec)


@numba.njit(cache=True, parallel=True)
def _fit_labeled_gaussian2d(
    data,
    x,
    y,
    labels,
    max_labels,
    amplitude,
    mean,
    covariance,
    precision,
    log_residual,
    label,
    sum_intensity,
    number_of_pixels,
):
    dim = 2
    q = 6
    k = amplitude.shape[2]
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            label_sums = np.zeros(max_labels + 1, dtype=np.float64)
            top_labels = np.zeros(k, dtype=np.uint16)
            top_sums = np.zeros(k, dtype=np.float64)

            for ii in range(data.shape[2]):
                for jj in range(data.shape[3]):
                    lab = labels[i, j, ii, jj]
                    value = data[i, j, ii, jj]
                    if lab > 0 and value > 0:
                        label_sums[lab] += value

            for lab in range(1, max_labels + 1):
                if label_sums[lab] > 0.0:
                    _insert_label_by_intensity(lab, label_sums[lab], top_labels, top_sums)

            for peak in range(k):
                lab = top_labels[peak]
                if lab == 0:
                    continue

                normal = np.zeros((10, 10), dtype=np.float64)
                rhs = np.zeros(10, dtype=np.float64)
                phi = np.zeros(10, dtype=np.float64)
                count = 0

                for ii in range(data.shape[2]):
                    for jj in range(data.shape[3]):
                        if labels[i, j, ii, jj] != lab:
                            continue
                        value = data[i, j, ii, jj]
                        if value == 0:
                            continue
                        x0 = x[ii, jj]
                        x1 = y[ii, jj]
                        phi[0] = 1.0
                        phi[1] = x0
                        phi[2] = x1
                        phi[3] = x0 * x0
                        phi[4] = x0 * x1
                        phi[5] = x1 * x1
                        _accumulate_normal_equations(
                            phi, q, np.log(float(value)), normal, rhs
                        )
                        count += 1

                label[i, j, peak] = lab
                sum_intensity[i, j, peak] = top_sums[peak]
                number_of_pixels[i, j, peak] = count

                if not _fit_labeled_region(
                    dim,
                    q,
                    count,
                    normal,
                    rhs,
                    amplitude[i, j, peak : peak + 1],
                    mean[i, j, peak],
                    covariance[i, j, peak],
                    precision[i, j, peak],
                ):
                    continue

                residual = 0.0
                for ii in range(data.shape[2]):
                    for jj in range(data.shape[3]):
                        if labels[i, j, ii, jj] != lab:
                            continue
                        value = data[i, j, ii, jj]
                        if value == 0:
                            continue
                        phi[0] = x[ii, jj]
                        phi[1] = y[ii, jj]
                        diff = np.log(float(value)) - _gaussian_log_value(
                            dim,
                            phi,
                            amplitude[i, j, peak : peak + 1],
                            mean[i, j, peak],
                            precision[i, j, peak],
                        )
                        residual += diff * diff
                log_residual[i, j, peak] = residual / count


@numba.njit(cache=True, parallel=True)
def _fit_labeled_gaussian3d(
    data,
    x,
    y,
    z,
    labels,
    max_labels,
    amplitude,
    mean,
    covariance,
    precision,
    log_residual,
    label,
    sum_intensity,
    number_of_pixels,
):
    dim = 3
    q = 10
    k = amplitude.shape[2]
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            label_sums = np.zeros(max_labels + 1, dtype=np.float64)
            top_labels = np.zeros(k, dtype=np.uint16)
            top_sums = np.zeros(k, dtype=np.float64)

            for ii in range(data.shape[2]):
                for jj in range(data.shape[3]):
                    for kk in range(data.shape[4]):
                        lab = labels[i, j, ii, jj, kk]
                        value = data[i, j, ii, jj, kk]
                        if lab > 0 and value > 0:
                            label_sums[lab] += value

            for lab in range(1, max_labels + 1):
                if label_sums[lab] > 0.0:
                    _insert_label_by_intensity(lab, label_sums[lab], top_labels, top_sums)

            for peak in range(k):
                lab = top_labels[peak]
                if lab == 0:
                    continue

                normal = np.zeros((10, 10), dtype=np.float64)
                rhs = np.zeros(10, dtype=np.float64)
                phi = np.zeros(10, dtype=np.float64)
                count = 0

                for ii in range(data.shape[2]):
                    for jj in range(data.shape[3]):
                        for kk in range(data.shape[4]):
                            if labels[i, j, ii, jj, kk] != lab:
                                continue
                            value = data[i, j, ii, jj, kk]
                            if value == 0:
                                continue
                            x0 = x[ii, jj, kk]
                            x1 = y[ii, jj, kk]
                            x2 = z[ii, jj, kk]
                            phi[0] = 1.0
                            phi[1] = x0
                            phi[2] = x1
                            phi[3] = x2
                            phi[4] = x0 * x0
                            phi[5] = x0 * x1
                            phi[6] = x0 * x2
                            phi[7] = x1 * x1
                            phi[8] = x1 * x2
                            phi[9] = x2 * x2
                            _accumulate_normal_equations(
                                phi, q, np.log(float(value)), normal, rhs
                            )
                            count += 1

                label[i, j, peak] = lab
                sum_intensity[i, j, peak] = top_sums[peak]
                number_of_pixels[i, j, peak] = count

                if not _fit_labeled_region(
                    dim,
                    q,
                    count,
                    normal,
                    rhs,
                    amplitude[i, j, peak : peak + 1],
                    mean[i, j, peak],
                    covariance[i, j, peak],
                    precision[i, j, peak],
                ):
                    continue

                residual = 0.0
                for ii in range(data.shape[2]):
                    for jj in range(data.shape[3]):
                        for kk in range(data.shape[4]):
                            if labels[i, j, ii, jj, kk] != lab:
                                continue
                            value = data[i, j, ii, jj, kk]
                            if value == 0:
                                continue
                            phi[0] = x[ii, jj, kk]
                            phi[1] = y[ii, jj, kk]
                            phi[2] = z[ii, jj, kk]
                            diff = np.log(float(value)) - _gaussian_log_value(
                                dim,
                                phi,
                                amplitude[i, j, peak : peak + 1],
                                mean[i, j, peak],
                                precision[i, j, peak],
                            )
                            residual += diff * diff
                log_residual[i, j, peak] = residual / count


@numba.njit(cache=True, parallel=True)
def _fit_gaussian3d(data, x, y, z, amplitude, mean, covariance, precision, log_residual):
    dim = 3
    q = 10
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            normal = np.zeros((10, 10), dtype=np.float64)
            rhs = np.zeros(10, dtype=np.float64)
            phi = np.zeros(10, dtype=np.float64)
            count = 0

            for ii in range(data.shape[2]):
                for jj in range(data.shape[3]):
                    for kk in range(data.shape[4]):
                        value = data[i, j, ii, jj, kk]
                        if value == 0:
                            continue

                        x0 = x[ii, jj, kk]
                        x1 = y[ii, jj, kk]
                        x2 = z[ii, jj, kk]
                        phi[0] = 1.0
                        phi[1] = x0
                        phi[2] = x1
                        phi[3] = x2
                        phi[4] = x0 * x0
                        phi[5] = x0 * x1
                        phi[6] = x0 * x2
                        phi[7] = x1 * x1
                        phi[8] = x1 * x2
                        phi[9] = x2 * x2
                        _accumulate_normal_equations(
                            phi, q, np.log(float(value)), normal, rhs
                        )
                        count += 1

            if not _finish_fit(
                dim,
                q,
                count,
                normal,
                rhs,
                amplitude[i, j : j + 1],
                mean[i, j],
                covariance[i, j],
                precision[i, j],
            ):
                _reset_fit(
                    dim,
                    amplitude[i, j : j + 1],
                    mean[i, j],
                    covariance[i, j],
                    precision[i, j],
                    log_residual[i, j : j + 1],
                )
                continue

            residual = 0.0
            for ii in range(data.shape[2]):
                for jj in range(data.shape[3]):
                    for kk in range(data.shape[4]):
                        value = data[i, j, ii, jj, kk]
                        if value == 0:
                            continue
                        phi[0] = x[ii, jj, kk]
                        phi[1] = y[ii, jj, kk]
                        phi[2] = z[ii, jj, kk]
                        diff = np.log(float(value)) - _gaussian_log_value(
                            dim,
                            phi,
                            amplitude[i, j : j + 1],
                            mean[i, j],
                            precision[i, j],
                        )
                        residual += diff * diff
            log_residual[i, j] = residual / count
