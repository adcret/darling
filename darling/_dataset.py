import time

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.ndimage
from matplotlib.colors import hsv_to_rgb

import darling


class _Visualizer(object):

    # TODO: some of this should probably be in the properties module...

    def __init__(self, dset_reference):
        self.dset = dset_reference
        self.labels = self.dset.reader.motor_names[:]
        for i, label in enumerate(self.labels):
            if "chi" in self.labels[i]:
                self.labels[i] = r"$\chi$"
            if "phi" in self.labels[i]:
                self.labels[i] = r"$\phi$"
            if "ccmth" in self.labels[i]:
                self.labels[i] = r"ccmth"
            if "strain" in self.labels[i]:
                self.labels[i] = r"$\varepsilon$"
            if "energy" in self.labels[i]:
                self.labels[i] = r"energy"
            if "diffrz" in self.labels[i]:
                self.labels[i] = r"diffrz"
            if "diffry" in self.labels[i]:
                self.labels[i] = r"diffry"
        self.motor_xlabel = self.labels[0]
        self.motor_ylabel = self.labels[1]

        self.xlabel = "Detector row index"
        self.ylabel = "Detector column index"

    def mean(self):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 2, figsize=(9, 6), sharex=True, sharey=True)
        fig.suptitle("Mean Map \nfirst moment around motor coordinates", fontsize=22)
        im_ratio = self.dset.mean.shape[0] / self.dset.mean.shape[1]
        for i in range(2):
            im = ax[i].imshow(self.dset.mean[:, :, i], cmap="jet")
            fig.colorbar(im, ax=ax[i], fraction=0.046 * im_ratio, pad=0.04)
            ax[i].set_title("Mean in motor " + self.labels[i], fontsize=14)
            ax[i].set_xlabel(self.xlabel, fontsize=14)
            if i == 0:
                ax[i].set_ylabel(self.ylabel, fontsize=14)
        plt.tight_layout()
        plt.show()

    def covariance(self):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
        fig.suptitle(
            "Covariance Map \nsecond moment around motor coordinates", fontsize=22
        )
        im_ratio = self.dset.covariance.shape[0] / self.dset.covariance.shape[1]
        for i in range(2):
            for j in range(2):
                im = ax[i, j].imshow(self.dset.covariance[:, :, i, j], cmap="magma")
                fig.colorbar(im, ax=ax[i, j], fraction=0.046 * im_ratio, pad=0.04)
                ax[i, j].set_title(
                    "Covar[" + self.labels[i] + ", " + self.labels[j] + "]", fontsize=14
                )
                if j == 0:
                    ax[i, j].set_ylabel(self.ylabel, fontsize=14)
                if i == 1:
                    ax[i, j].set_xlabel(self.xlabel, fontsize=14)
        plt.tight_layout()
        plt.show()

    def misorientation(self):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(9, 9), sharex=True, sharey=True)
        fig.suptitle(
            "Misorientation Map \nL2 norm of mean map after median subtraction",
            fontsize=22,
        )
        mean = self.dset.mean.copy()
        mean[:, :, 0] -= np.median(mean[:, :, 0].flatten())
        mean[:, :, 1] -= np.median(mean[:, :, 1].flatten())
        misori = np.linalg.norm(mean, axis=-1)
        im_ratio = misori.shape[0] / misori.shape[1]
        im = ax.imshow(misori, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046 * im_ratio, pad=0.04)
        ax.set_title("Misorientation", fontsize=14)
        ax.set_xlabel(self.xlabel, fontsize=14)
        ax.set_ylabel(self.ylabel, fontsize=14)
        plt.tight_layout()
        plt.show()

    def _wrap2pi(self, x):
        """
        Python implementation of Matlab method `wrapTo2pi`.
        Wraps angles in x, in radians, to the interval [0, 2*pi] such that 0 maps
        to 0 and 2*pi maps to 2*pi. In general, positive multiples of 2*pi map to
        2*pi and negative multiples of 2*pi map to 0.
        """
        xwrap = np.remainder(x - np.pi, 2 * np.pi)
        mask = np.abs(xwrap) > np.pi
        xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
        return xwrap + np.pi

    def _hsv_key(self, angles, radius):
        return np.stack(
            (
                angles,  # HUE (the actual color)
                radius,  # SATURATION (how saturated the color is)
                np.ones(angles.shape),  # VALUE. (white to black)
            ),
            axis=2,
        )

    def _mosa(self, ang1, ang2):
        angles = np.arctan2(-ang1, -ang2)
        anlges_normalized = self._wrap2pi(angles) / np.pi / 2
        radius = np.sqrt(ang2**2 + ang2**2)
        radius_normalized = radius / radius.max()
        return anlges_normalized, radius_normalized

    def _hsv_colormap(self):
        ang_grid = np.linspace(-1, 1, 400)
        ang1, ang2 = np.meshgrid(ang_grid, ang_grid)
        angles, radius = self._mosa(ang1, ang2)
        hsv_key = self._hsv_key(angles, radius)
        colormap = hsv_to_rgb(hsv_key)
        return colormap

    def mosaicity(self):

        # Calculate Mosa Imager
        mean = self.dset.mean.copy()
        ranges = np.array(
            [
                [mean[:, :, 0].min(), mean[:, :, 0].max()],
                [mean[:, :, 1].min(), mean[:, :, 1].max()],
            ]
        )
        ranges_magnitude = [ranges[0, 1] - ranges[0, 0], ranges[1, 1] - ranges[1, 0]]
        chi_norm = (mean[:, :, 0] - mean[:, :, 0].min()) / ranges_magnitude[0] - 0.5
        phi_norm = (mean[:, :, 1] - mean[:, :, 1].min()) / ranges_magnitude[1] - 0.5
        angles, radius = self._mosa(chi_norm, phi_norm)
        hsv_key = self._hsv_key(angles, radius)

        mosa = hsv_to_rgb(hsv_key)
        colormap = self._hsv_colormap()

        plt.style.use("dark_background")
        fig, axs = plt.subplots(
            1, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [3, 1]}
        )
        fig.suptitle(
            "Mosaicity Map \n maps motors to a cylindrical HSV colorspace",
            fontsize=22,
        )
        axs[0].imshow(mosa)
        axs[0].set_title(r"Mosaicity Map", fontsize=14)
        axs[0].set_xlabel(self.xlabel, fontsize=14)
        axs[0].set_ylabel(self.ylabel, fontsize=14)
        axs[1].imshow(colormap)
        axs[1].set_xlabel(self.motor_xlabel, fontsize=14)
        axs[1].set_ylabel(self.motor_ylabel, fontsize=14)
        axs[1].set_title(r"Color Map", fontsize=14)
        chiTicks = np.linspace(0, colormap.shape[1] - 1, 5)
        chi_label = np.linspace(ranges[0, 0], ranges[0, 1], 5)
        chi_label = np.round(chi_label, decimals=3)
        chi_label = np.array([f"{chi:.3f}" for chi in chi_label])
        axs[1].set_xticks(chiTicks)
        axs[1].set_xticklabels(chi_label)

        phiTicks = np.linspace(0, colormap.shape[1] - 1, 5)
        phi_label = np.linspace(ranges[1, 0], ranges[1, 1], 5)
        phi_label = np.round(phi_label, decimals=3)
        phi_label = np.array([f"{phi:.3f}" for phi in phi_label])

        axs[1].set_yticks(phiTicks)
        axs[1].set_yticklabels(phi_label)

        plt.tight_layout()
        plt.show()


class DataSet(object):
    """A DFXM data-set.

    This is the master data class of darling. Given a reader the DataSet class will read data form
    arbitrary layers, process, threshold, compute moments, visualize results, and compile 3D feature maps.

    Args:
        reader (:obj: `darling.reader.Reader`): A file reader implementing, at least, the functionallity
            specified in darling.reader.Reader().

    Attributes:
        reader (:obj: `darling.reader.Reader`): A file reader implementing, at least, the functionallity
            specified in darling.reader.Reader().

    """

    def __init__(self, reader):
        self.reader = reader
        self.plot = _Visualizer(self)
        self.mean, self.covariance = None, None
        self.mean_3d, self.covariance_3d = None, None

    def load_scan(self, args, scan_id, roi=None):
        """Load a scan into RM.

        NOTE: Input args should match the darling.reader.Reader used, however it was implemented.

        Args:
            args, (:obj: `tuple` or other): Depending on the reader implementation this is either
                a tuple of arguments in which case the reader is called as: self.reader(\*args, scan_id, roi)
                or, alternatively, this is a single argument in which case the reader is called 
                as self.reader(args, scan_id, roi) the provided reader must be compatible with one
                of these call signatures.
            scan_id (:obj:`str`): scan id to load from, these are internal keys to diffirentiate
                layers.
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded. The roi refers to the detector
                dimensions.

        """
        if isinstance(args, tuple):
            self.data, self.motors = self.reader(*args, scan_id, roi)
        else:
            self.data, self.motors = self.reader(args, scan_id, roi)

    def subtract(self, value):
        """Subtract a fixed integer value form the data. Protects against uint16 sign flips.

        Args:
            value (:obj:`int`): value to subtract.

        """
        self.data.clip(value, None, out=self.data)
        self.data -= value

    def estimate_background(self):
        """Automatic background correction based on image statistics.

        a set of sample data is extracted from the data block. The median and standard deviations are iteratively
        fitted, rejecting outliers (which here is diffraction signal). Once the noise distirbution has been established
        the value corresponding to the 99.99% percentile is returned. I.e the far tail of the noise is returned.

        """
        sample_size = 40000
        index = np.random.permutation(sample_size)
        sample = self.data.flat[index]
        sample = np.sort(sample)
        noise = sample.copy()
        for i in range(20):
            mu = np.median(noise)
            std = np.std(noise)
            noise = noise[np.abs(noise) < mu + 2 * 3.891 * std]  # 99.99% confidence
        background = np.max(noise)
        return background

    def moments(self):
        """Compute first and second moments.

        The internal attributes self.mean and self.covariance are set when this function is run.

        Returns:
            (:obj:`tupe` of :obj:`numpy array`): mean and covariance maps of shapes (a,b,2) and (a,b,2,2)
                respectively with a=self.data.shape[0] and b=self.data.shape[1].
        """
        self.mean, self.covariance = darling.properties.moments(self.data, self.motors)
        return self.mean, self.covariance

    def integrate(self):
        """Return the summed data stack along the motor dimensions avoiding data stack copying.

        Returns:
            (:obj:`numpy array`): integrated frames, a 2D numpy array.
        """
        out = np.zeros(
            (self.data.shape[0], self.data.shape[1]), np.float32
        )  # avoid casting and copying.
        integrated_frames = np.sum(self.data, axis=(2, 3), out=out)
        return integrated_frames

    def estimate_mask(
        self,
        threshold=200,
        erosion_iterations=3,
        dilation_iterations=25,
        fill_holes=True,
    ):
        """Segment the sample diffracting region based on summed intensity along motor dimensions.

        Args:
            threshold (:obj:`int`):  a summed count value above which the sample is defined.
            erosion_iterations (:obj:`int`): Number of times to erode the mask using a 2,2 structure.
            dilation_iterations (:obj:`int`): Number of times to dilate the mask using a 2,2 structure.
            fill_holes (:obj:`bool`):  Fill enclosed holes in the final mask.

        Returns:
            (:obj:`numpy array`): Returns: a binary 2D mask of the sample.

        """
        mask = self.integrate() > threshold
        mask = scipy.ndimage.binary_erosion(
            mask, structure=np.ones((2, 2)), iterations=erosion_iterations
        )
        mask = scipy.ndimage.binary_dilation(
            mask, structure=np.ones((2, 2)), iterations=dilation_iterations
        )
        if fill_holes:
            mask = scipy.ndimage.binary_fill_holes(mask)
        return mask

    def compile_layers(
        self, reader_args, scan_ids, threshold=None, roi=None, verbose=False
    ):
        """Sequentially load a series of scans and assemble the 3D moment maps.

        this loads the mosa data array with shape a,b,m,n,(o) where a,b are the detector dimension and
        m,n,(o) are the motor dimensions as ordered in the `self.motor_names`.

        NOTE: This function will load data sequentially and compute moments on the fly. While all
        moment maps are stored and concatenated, only one scan (the raw 4d or 5d data) is kept in
        memory at a time to enhance RAM performance.

        Args:
            data_name (:obj:`str`): path to the data (in the h5) without the prepended scan id
            scan_ids (:obj:`str`): scan ids to load, e.g 1.1, 2.1 etc...
            threshold (:obj:`int` or :obj:`str`): background subtraction value or string 'auto' in which
                case a default background estimation is performed and subtracted. Defaults to None, in which
                case no background is subtracted.
            roi (:obj:`tuple` or :obj:`int`): row_min row_max and column_min and column_max, defaults to None, 
                in which case all data is loaded 
            verbose (:obj:`bool`): Print loading progress or not.

        """
        mean_3d = []
        covariance_3d = []
        tot_time = 0
        for i, scan_id in enumerate(scan_ids):

            t1 = time.perf_counter()

            if verbose:
                print(
                    "\nREADING SCAN: "
                    + str(i + 1)
                    + " out of totally "
                    + str(len(scan_ids))
                    + " scans"
                )
            self.load_scan(reader_args, scan_id, roi)

            if threshold is not None:
                if threshold == "auto":
                    if verbose:
                        print(
                            "    Subtracting estimated background for scan id "
                            + str(scan_id)
                            + " ..."
                        )
                    _threshold = self.estimate_background()
                    self.threshold(_threshold)
                else:
                    if verbose:
                        print(
                            "    Subtracting fixed background value="
                            + str(threshold)
                            + " for scan id "
                            + str(scan_id)
                            + " ..."
                        )
                    self.threshold(threshold)

            if verbose:
                print("    Computing moments for scan id " + str(scan_id) + " ...")

            mean, covariance = self.moments()

            if verbose:
                print(
                    "    Concatenating to 3D volume for scan id "
                    + str(scan_id)
                    + " ..."
                )
            mean_3d.append(mean)
            covariance_3d.append(covariance)

            t2 = time.perf_counter()
            tot_time += t2 - t1

            estimated_time_left = str((tot_time / (i + 1)) * (len(scan_ids) - i - 1))
            if verbose:
                print("    Estimated time left is : " + estimated_time_left + " s")

        self.mean_3d = np.array(mean_3d)
        self.covariance_3d = np.array(covariance_3d)

        if verbose:
            print("\ndone! Total time was : " + str(tot_time) + " s")

        return self.mean_3d, self.covariance_3d

    def to_paraview(self, file):
        """Write moment maps to paraview readable format for 3D visualisation.

        The written data array will have attributes as:

            cov_11, cov_12, (cov_13), cov_22, (cov_23, cov_33) : Elements of covariance matrix.
            mean_1, mean_2, (mean_3) : The first moments in each dimension.

        Here 1 signifies the self.motors[0] dimension while 2 is in self.motors[2], (and
        3 in self.motors[3], when the scan is 3D)

        NOTE: Requires that 3D moment maps have been compiled via compile_layers().

        Args:
            file (:obj:`string`): Absolute path ending with desired filename.

        """

        dim = np.array(self.mean_3d.shape)[-1]
        s, a, b = np.array(self.mean_3d.shape)[0:3]
        sg = np.linspace(0, s, s)
        ag = np.linspace(0, a, a)
        bg = np.linspace(0, b, b)
        mesh = np.meshgrid(sg, ag, bg, indexing="ij")
        points = np.array([x.flatten() for x in mesh])
        N = points.shape[1]
        cells = [("vertex", np.array([[i] for i in range(N)]))]

        if len(file.split(".")) == 1:
            filename = file + ".xdmf"
        else:
            filename = file

        point_data = {}
        for i in range(dim):
            point_data["mean_" + str(i + 1)] = self.mean_3d[:, :, :, i].flatten()
            for j in range(i, dim):
                point_data["cov_" + str(i + 1) + str(j + 1)] = self.covariance_3d[
                    :, :, :, i, j
                ].flatten()

        meshio.Mesh(
            points.T,
            cells,
            point_data=point_data,
        ).write(filename)


if __name__ == "__main__":
    pass
