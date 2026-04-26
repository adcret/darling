import unittest

import numpy as np

import darling


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

        fit = darling.fitting.gaussian(data, coordinates)

        self.assertEqual(fit["amplitude"].shape, data.shape[:2])
        self.assertEqual(fit["mean"].shape, (*data.shape[:2], 2))
        self.assertEqual(fit["covariance"].shape, (*data.shape[:2], 2, 2))
        self.assertEqual(fit["precision"].shape, (*data.shape[:2], 2, 2))
        for value in fit.values():
            self.assertEqual(value.dtype, np.float32)

        np.testing.assert_allclose(fit["amplitude"], amplitude, rtol=0.01)
        np.testing.assert_allclose(fit["mean"], mean, atol=0.002)
        np.testing.assert_allclose(fit["covariance"], covariance, atol=0.002)
        np.testing.assert_allclose(
            fit["precision"], np.linalg.inv(covariance), atol=0.02
        )
        np.testing.assert_array_less(fit["log_residual"], 1e-7)

        fit_from_properties = darling.properties.gaussian_fit(data, coordinates)
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

        fit = darling.fitting.gaussian(data, coordinates)

        self.assertEqual(fit["amplitude"].shape, data.shape[:2])
        self.assertEqual(fit["mean"].shape, (*data.shape[:2], 3))
        self.assertEqual(fit["covariance"].shape, (*data.shape[:2], 3, 3))
        self.assertEqual(fit["precision"].shape, (*data.shape[:2], 3, 3))

        np.testing.assert_allclose(fit["amplitude"], amplitude, rtol=0.01)
        np.testing.assert_allclose(fit["mean"], mean, atol=0.002)
        np.testing.assert_allclose(fit["covariance"], covariance, atol=0.002)
        np.testing.assert_allclose(
            fit["precision"], np.linalg.inv(covariance), atol=0.02
        )
        np.testing.assert_array_less(fit["log_residual"], 1e-7)

    def test_gaussian_fit_rejects_unsupported_dimensions(self):
        data = np.ones((2, 2, 5), dtype=np.uint16)
        coordinates = np.arange(5, dtype=np.float32).reshape(1, 5)
        with self.assertRaises(ValueError):
            darling.fitting.gaussian(data, coordinates)


if __name__ == "__main__":
    unittest.main()
