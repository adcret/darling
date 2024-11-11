import numpy as np
import matplotlib.pyplot as plt


class DataSet(object):

    def __init__(self, reader):
        self.reader = reader

    def threshold(self, threshold):
        """Threshold the data based on a fixed value.

        Args:
            threshold, int : threshold value compared to individual pixels.

        """
        self.data.clip(threshold, None, out=self.data)
        self.data-=threshold
        
    def moments(self):
        m1_mesh,m2_mesh  = np.meshgrid(*self.motors, indexing='ij')
        points = np.array([m1_mesh.flatten(), m2_mesh.flatten()])
        
        t1 = time.perf_counter()
        dum = np.arange(2).astype(np.float32)
        
        print("Computing first moments..")
        res = np.zeros((self.data.shape[0], self.data.shape[1], 2), dtype=np.float32)
        self.mean = _first_moments(self.data, m1_mesh, m2_mesh, dum, res)
            
        print("Computing second moments..")
        res = np.zeros((self.data.shape[0], self.data.shape[1], 2, 2), dtype=np.float32)
        self.covariance = _second_moments(self.data, self.mean, m1_mesh, m2_mesh, points, dum, res)
 
        t2 = time.perf_counter()
        print("execution time [s]: " + str(t2-t1))
    
    def get_mask(self, threshold=200):
        """Try to segment the sample based on summed intensity along motor dimensions.

        Args:
            threshold, int : value above which the sample is defined.

        Returns:
            Returns: a binary 2D maks of the sample.

        """
        mask = np.sum( self.data, axis=(2,3)) > threshold
        mask = binary_erosion(mask, structure=np.ones((2,2)), iterations=3)
        mask = binary_dilation(mask, structure=np.ones((2,2)), iterations=25)
        mask = binary_fill_holes(mask)
        return mask

    def compile_layers(self, data_name, threshold, scan_IDs=None , roi=None):
        """Sequentially load a series of scans and assemble the 3D moment maps.
        
        this loads the mosa data array with shape N,N,m,n where N is the detector dimension and
        m,n are the motor dimensions as ordered in the self.motor_names.
        
        Args:
            data_name, str : path to the data without the prepended scan id
            threshold, int : background subtraction value
            scan_IDs, str : scan ids to load, e.g 1.1, 2.1 etc...
            roi, tuple of int, row_min row_max and column_min and column_max, defaults to None, in which case all data is loaded
        """
        if scan_IDs is None:
            # find all scan ids. 
            with h5py.File(self.abs_path_to_h5_file, 'r') as h5f:
                scan_IDs = np.array([scan_ID for scan_ID in h5f.keys() if ".1" in scan_ID])
                scan_IDs_index_sorted = np.argsort([int(scan_ID.split(".")[0]) for scan_ID in scan_IDs])
                scan_IDs = scan_IDs[scan_IDs_index_sorted]
        layer_stack_mean = []
        layer_stack_covariance = []
        layer_positions = []
        for i, scan_id in enumerate(scan_IDs):
            print("read in scan "+str(i-1)+" of "+str(len(scan_IDs)-1))
            self.load_scan(data_name, scan_id, roi=roi)
            self.threshold(threshold)
            self.moments()
            
            layer_positions.append(self.c_motors[0][0])
            layer_stack_mean.append(self.mean)
            layer_stack_covariance.append(self.covariance)
        self.layer_stack_mean = np.array(layer_stack_mean)
        self.layer_stack_covariance = np.array(layer_stack_covariance)
        self.layer_positions = np.array(layer_positions)
        print("finished!")


if __name__=='__main__':
    import darling

    dset = darling.DataSet()