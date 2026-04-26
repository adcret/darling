import unittest

import numpy as np

import darling
from darling.properties.curvefit import fit_nd_gaussian


def _gaussian_values(coordinates, amplitude, mean, covariance):
    precision = np.linalg.inv(covariance)
    exponent = np.zeros_like(coordinates[0], dtype=np.float64)
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            exponent += (
                (coordinates[i] - mean[i])
                * precision[i, j]
                * (coordinates[j] - mean[j])
            )
    return amplitude * np.exp(-0.5 * exponent)


def _assert_fit_matches(testcase, fit, data_shape, mean, covariance, amplitude):
    dim = len(mean)
    testcase.assertEqual(fit["amplitude"].shape, data_shape[:2])
    testcase.assertEqual(fit["mean"].shape, (*data_shape[:2], dim))
    testcase.assertEqual(fit["covariance"].shape, (*data_shape[:2], dim, dim))
    testcase.assertEqual(fit["precision"].shape, (*data_shape[:2], dim, dim))
    for value in fit.values():
        testcase.assertEqual(value.dtype, np.float32)

    np.testing.assert_allclose(fit["amplitude"], amplitude, rtol=0.01)
    np.testing.assert_allclose(
        fit["mean"], np.broadcast_to(mean, fit["mean"].shape), atol=0.002
    )
    np.testing.assert_allclose(
        fit["covariance"],
        np.broadcast_to(covariance, fit["covariance"].shape),
        atol=0.002,
    )
    np.testing.assert_allclose(
        fit["precision"],
        np.broadcast_to(np.linalg.inv(covariance), fit["precision"].shape),
        atol=0.02,
    )
    np.testing.assert_array_less(fit["log_residual"], 1e-4)


def _sort_peak_axis(fit):
    order = np.argsort(fit["mean"][0, 0, :, 0])
    return {
        "amplitude": fit["amplitude"][..., order],
        "mean": fit["mean"][..., order, :],
        "covariance": fit["covariance"][..., order, :, :],
        "precision": fit["precision"][..., order, :, :],
        "log_residual": fit["log_residual"][..., order],
        "label": fit["label"][..., order],
        "sum_intensity": fit["sum_intensity"][..., order],
        "number_of_pixels": fit["number_of_pixels"][..., order],
    }


class TestGaussianFit(unittest.TestCase):
    def test_gaussian_fit_2d(self):
        x = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
        coordinates = np.array(np.meshgrid(x, y, indexing="ij"), dtype=np.float32)
        amplitude = 52000.0
        mean = np.array([0.12, -0.18], dtype=np.float64)
        covariance = np.array([[0.23, 0.04], [0.04, 0.18]], dtype=np.float64)

        image = _gaussian_values(coordinates, amplitude, mean, covariance)
        data = np.zeros((3, 4, *image.shape), dtype=np.uint16)
        data[...] = np.round(image).astype(np.uint16)

        fit = fit_nd_gaussian(data, coordinates)

        _assert_fit_matches(self, fit, data.shape, mean, covariance, amplitude)

        fit_from_properties = darling.properties.fit_nd_gaussian(data, coordinates)
        np.testing.assert_allclose(fit_from_properties["mean"], fit["mean"])

    def test_gaussian_fit_3d(self):
        x = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        z = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
        coordinates = np.array(np.meshgrid(x, y, z, indexing="ij"), dtype=np.float32)
        amplitude = 61000.0
        mean = np.array([0.08, -0.11, 0.16], dtype=np.float64)
        covariance = np.array(
            [[0.30, 0.04, -0.02], [0.04, 0.26, 0.03], [-0.02, 0.03, 0.22]],
            dtype=np.float64,
        )

        image = _gaussian_values(coordinates, amplitude, mean, covariance)
        data = np.zeros((2, 3, *image.shape), dtype=np.uint16)
        data[...] = np.round(image).astype(np.uint16)

        fit = fit_nd_gaussian(data, coordinates)

        _assert_fit_matches(self, fit, data.shape, mean, covariance, amplitude)

    def test_gaussian_fit_2d_labeled_regions(self):
        x = np.linspace(-1.0, 1.0, 25, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, 27, dtype=np.float32)
        coordinates = np.array(np.meshgrid(x, y, indexing="ij"), dtype=np.float32)
        amplitude = np.array([52000.0, 43000.0])
        mean = np.array([[-0.50, -0.42], [0.50, 0.42]], dtype=np.float64)
        covariance = np.array(
            [
                [[0.010, 0.001], [0.001, 0.012]],
                [[0.011, -0.001], [-0.001, 0.010]],
            ],
            dtype=np.float64,
        )

        components = []
        image = np.zeros_like(coordinates[0], dtype=np.float64)
        labels = np.zeros((2, 3, *image.shape), dtype=np.uint16)
        for label in range(2):
            component = _gaussian_values(
                coordinates, amplitude[label], mean[label], covariance[label]
            )
            component[component < 1] = 0
            components.append(component)
            image += component
        component_stack = np.stack(components)
        label_image = np.argmax(component_stack, axis=0).astype(np.uint16) + 1
        label_image[np.max(component_stack, axis=0) == 0] = 0
        labels[...] = label_image
        data = np.zeros((2, 3, *image.shape), dtype=np.uint16)
        data[...] = np.round(image).astype(np.uint16)

        fit = _sort_peak_axis(fit_nd_gaussian(data, coordinates, labels=labels, k=2))

        self.assertEqual(fit["amplitude"].shape, (*data.shape[:2], 2))
        self.assertEqual(fit["mean"].shape, (*data.shape[:2], 2, 2))
        self.assertEqual(fit["covariance"].shape, (*data.shape[:2], 2, 2, 2))
        for peak in range(2):
            np.testing.assert_allclose(
                fit["amplitude"][..., peak], amplitude[peak], rtol=0.03
            )
            np.testing.assert_allclose(
                fit["mean"][..., peak, :],
                np.broadcast_to(mean[peak], fit["mean"][..., peak, :].shape),
                atol=0.003,
            )
            np.testing.assert_allclose(
                fit["covariance"][..., peak, :, :],
                np.broadcast_to(
                    covariance[peak], fit["covariance"][..., peak, :, :].shape
                ),
                atol=0.003,
            )
        np.testing.assert_array_less(fit["log_residual"], 1e-2)

    def test_gaussian_fit_2d_local_max_regions(self):
        x = np.linspace(-1.0, 1.0, 25, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, 27, dtype=np.float32)
        coordinates = np.array(np.meshgrid(x, y, indexing="ij"), dtype=np.float32)
        means = np.array([[-0.42, -0.28], [0.38, 0.31]], dtype=np.float64)
        covariance = np.array([[0.018, 0.0], [0.0, 0.018]], dtype=np.float64)
        image = (
            _gaussian_values(coordinates, 52000.0, means[0], covariance)
            + _gaussian_values(coordinates, 43000.0, means[1], covariance)
        )
        image[image < 20] = 0
        data = np.zeros((1, 1, *image.shape), dtype=np.uint16)
        data[0, 0] = np.round(image).astype(np.uint16)

        fit = fit_nd_gaussian(data, coordinates, mode="local_max", k=2)
        fitted_means = fit["mean"][0, 0, np.argsort(fit["mean"][0, 0, :, 0])]
        np.testing.assert_allclose(fitted_means, means, atol=0.01)
        np.testing.assert_array_equal(fit["number_of_pixels"][0, 0] > 0, [True, True])

    def test_gaussian_fit_3d_labeled_regions(self):
        x = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, 15, dtype=np.float32)
        z = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
        coordinates = np.array(np.meshgrid(x, y, z, indexing="ij"), dtype=np.float32)
        amplitude = np.array([61000.0, 47000.0])
        mean = np.array([[-0.45, -0.42, -0.34], [0.45, 0.42, 0.34]], dtype=np.float64)
        covariance = np.array(
            [
                [[0.016, 0.001, 0.001], [0.001, 0.015, 0.000], [0.001, 0.000, 0.014]],
                [[0.015, 0.000, 0.001], [0.000, 0.017, 0.001], [0.001, 0.001, 0.016]],
            ],
            dtype=np.float64,
        )
        components = []
        image = np.zeros_like(coordinates[0], dtype=np.float64)
        labels = np.zeros((1, 2, *image.shape), dtype=np.uint16)
        for label in range(2):
            component = _gaussian_values(
                coordinates, amplitude[label], mean[label], covariance[label]
            )
            component[component < 1] = 0
            components.append(component)
            image += component
        component_stack = np.stack(components)
        label_image = np.argmax(component_stack, axis=0).astype(np.uint16) + 1
        label_image[np.max(component_stack, axis=0) == 0] = 0
        labels[...] = label_image
        data = np.zeros((1, 2, *image.shape), dtype=np.uint16)
        data[...] = np.round(image).astype(np.uint16)

        fit = _sort_peak_axis(fit_nd_gaussian(data, coordinates, labels=labels, k=2))
        self.assertEqual(fit["mean"].shape, (*data.shape[:2], 2, 3))
        for peak in range(2):
            np.testing.assert_allclose(
                fit["amplitude"][..., peak], amplitude[peak], rtol=0.03
            )
            np.testing.assert_allclose(
                fit["mean"][..., peak, :],
                np.broadcast_to(mean[peak], fit["mean"][..., peak, :].shape),
                atol=0.004,
            )
            np.testing.assert_allclose(
                fit["covariance"][..., peak, :, :],
                np.broadcast_to(
                    covariance[peak], fit["covariance"][..., peak, :, :].shape
                ),
                atol=0.004,
            )
        np.testing.assert_array_less(fit["log_residual"], 1e-2)

    def test_gaussian_fit_3d_local_max_regions(self):
        x = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, 15, dtype=np.float32)
        z = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
        coordinates = np.array(np.meshgrid(x, y, z, indexing="ij"), dtype=np.float32)
        means = np.array([[-0.45, -0.42, -0.34], [0.45, 0.42, 0.34]], dtype=np.float64)
        covariance = np.diag([0.016, 0.016, 0.016])
        image = (
            _gaussian_values(coordinates, 61000.0, means[0], covariance)
            + _gaussian_values(coordinates, 47000.0, means[1], covariance)
        )
        image[image < 20] = 0
        data = np.zeros((1, 1, *image.shape), dtype=np.uint16)
        data[0, 0] = np.round(image).astype(np.uint16)

        fit = fit_nd_gaussian(data, coordinates, mode="local_max", k=2)
        fitted_means = fit["mean"][0, 0, np.argsort(fit["mean"][0, 0, :, 0])]
        np.testing.assert_allclose(fitted_means, means, atol=0.02)
        np.testing.assert_array_equal(fit["number_of_pixels"][0, 0] > 0, [True, True])

    def test_gaussian_fit_rejects_unsupported_dimensions(self):
        data = np.ones((2, 2, 5), dtype=np.uint16)
        coordinates = np.arange(5, dtype=np.float32).reshape(1, 5)
        with self.assertRaises(ValueError):
            fit_nd_gaussian(data, coordinates)


if __name__ == "__main__":
    unittest.main()
