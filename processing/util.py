import numpy
import zipfile

import math as m
import numpy as np
# np.random.seed(123)
import scipy.io
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

class FileReader(object):
    """
    Read one or multiple numpy arrays from a text/bz2 file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_stream = file_path

    def read_array(self, dtype=numpy.float64, skip_rows=0, use_cols=None, matlab_data_name=None):
        try:
            # Try to read H5:
            if self.file_path.endswith('.h5'):
                return numpy.array([])

            # Try to read NumPy:
            if self.file_path.endswith('.txt') or self.file_path.endswith('.bz2'):
                return self._read_text(self.file_stream, dtype, skip_rows, use_cols)

            if self.file_path.endswith('.npz') or self.file_path.endswith(".npy"):
                return numpy.load(self.file_stream)

            # Try to read Matlab format:
            return self._read_matlab(self.file_stream, matlab_data_name)

        except Exception:
            raise ReaderException("Could not read from %s file" % self.file_path)


    def _read_text(self, file_stream, dtype, skip_rows, use_cols):

        array_result = numpy.loadtxt(file_stream, dtype=dtype, skiprows=skip_rows, usecols=use_cols)
        return array_result


    def _read_matlab(self, file_stream, matlab_data_name=None):

        if self.file_path.endswith(".mtx"):
            return scipy_io.mmread(file_stream)

        if self.file_path.endswith(".mat"):
            matlab_data = scipy_io.matlab.loadmat(file_stream)
            return matlab_data[matlab_data_name]


    def read_gain_from_brainstorm(self):

        if not self.file_path.endswith('.mat'):
            raise ReaderException("Brainstorm format is expected in a Matlab file not %s" % self.file_path)

        mat = scipy_io.loadmat(self.file_stream)
        expected_fields = ['Gain', 'GridLoc', 'GridOrient']

        for field in expected_fields:
            if field not in mat.keys():
                raise ReaderException("Brainstorm format is expecting field %s" % field)

        gain, loc, ori = (mat[field] for field in expected_fields)
        return (gain.reshape((gain.shape[0], -1, 3)) * ori).sum(axis=-1)

class ZipReader(object):
    """
    Read one or many numpy arrays from a ZIP archive.
    """

    def __init__(self, zip_path):
        self.zip_archive = zipfile.ZipFile(zip_path)

    def read_array_from_file(self, file_name, dtype=numpy.float64, skip_rows=0, use_cols=None, matlab_data_name=None):

        matching_file_name = None
        for actual_name in self.zip_archive.namelist():
            if file_name in actual_name and not actual_name.startswith("__MACOSX"):
                matching_file_name = actual_name
                break

        if matching_file_name is None:
            raise ReaderException("File %r not found in ZIP." % file_name)

        zip_entry = self.zip_archive.open(matching_file_name, 'r')

        if matching_file_name.endswith(".bz2"):
            temp_file = copy_zip_entry_into_temp(zip_entry, matching_file_name)
            file_reader = FileReader(temp_file)
            result = file_reader.read_array(dtype, skip_rows, use_cols, matlab_data_name)
            os.remove(temp_file)
            return result

        file_reader = FileReader(matching_file_name)
        file_reader.file_stream = zip_entry
        return file_reader.read_array(dtype, skip_rows, use_cols, matlab_data_name)


    def read_optional_array_from_file(self, file_name, dtype=numpy.float64, skip_rows=0,
                                      use_cols=None, matlab_data_name=None):
        try:
            return self.read_array_from_file(file_name, dtype, skip_rows, use_cols, matlab_data_name)
        except ReaderException:
            return numpy.array([])

'''
A Suite of utility functions for preprocessing wrapped within
a class
'''

class DataHandler(object):
    def __init__(self, data=None, labels=None):
        self.data = data
        self.labels = labels

    def reformatinput(self, data: np.ndarray, indices: list):
        '''
        Receives the the indices for train and test datasets.
        Outputs the train, validation, and test data and label datasets.

        Parameters:
        data            (np.ndarray) of [n_samples, n_colors, W, H], or
                        [n_timewindows, n_samples, n_colors, W, H] for time dependencies
        indices         (list of tuples) of indice tuples to include in training,
                        validation, testing.
                        indices[0] = train
                        indices[1] = test

        Output:
        (list) of tuples for training, validation, testing 
        '''
        # get train and test indices
        trainIndices = indices[0][len(indices[1]):].astype(np.int32)
        validIndices = indices[0][:len(indices[1])].astype(np.int32) # use part of training for validation
        testIndices = indices[1].astype(np.int32)

        # gets train, valid, test labels as int32
        trainlabels = np.squeeze(indices[trainIndices]).astype(np.int32)
        validlabels = np.squeeze(indices[validIndices]).astype(np.int32)
        testlabels = np.squeeze(indices[testIndices]).astype(np.int32)

        # Shuffling training data
        # shuffledIndices = np.random.permutation(len(trainIndices))
        # trainIndices = trainIndices[shuffledIndices]

        # get the data tuples for train, valid, test by slicing thru n_samples
        if data.ndim == 4:
            return [(data[trainIndices], trainlabels),
                    (data[validIndices], validlabels),
                    (data[testIndices], testlabels)]
        elif data.ndim == 5:
            return [(data[:, trainIndices], trainlabels),
                    (data[:, validIndices], validlabels),
                    (data[:, testIndices], testlabels)]

    def load_mat_data(self, data_file: str):
        '''
        Loads the data from MAT file. MAT file should contain two
        variables. 'featMat' which contains the feature matrix in the
        shape of [samples, features] and 'labels' which contains the output
        labels as a vector. Label numbers are assumed to start from 1.

        Parameters
        ----------
        data_file       (str) for the fullpath to the file

        Returns
        -------
        data: array_like
        '''
        print("Loading data from %s" % (data_file))

        dataMat = scipy.io.loadmat(data_file, mat_dtype=True)

        print("Data loading complete. Shape is %r" % (dataMat['featMat'].shape,))
        return dataMat['features'][:, :-1], dataMat['features'][:, -1] - 1   # Sequential indices

    def load_mat_locs(self, datafile: str):
        # '../Sample data/Neuroscan_locs_orig.mat'
        locs = scipy.io.loadmat(datafile)
        return locs

    def cart2sph(self, x: float, y: float, z: float):
        '''
        Transform Cartesian coordinates to spherical
        
        Paramters:
        x           (float) X coordinate
        y           (float) Y coordinate
        z           (float) Z coordinate

        :return: radius, elevation, azimuth
        '''
        x2_y2 = x**2 + y**2
        r = m.sqrt(x2_y2 + z**2)                    # r
        elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
        az = m.atan2(y, x)                          # Azimuth
        return r, elev, az

    def pol2cart(self, theta: float, rho: float):
        '''
        Transform polar coordinates to Cartesian

        Parameters
        theta          (float) angle value
        rho            (float) radius value

        :return: X, Y
        '''
        return rho * m.cos(theta), rho * m.sin(theta)

    def azim_proj(self, pos: list):
        '''
        Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
        Imagine a plane being placed against (tangent to) a globe. If
        a light source inside the globe projects the graticule onto
        the plane the result would be a planar, or azimuthal, map
        projection.

        Parameters:
        pos         (list) positions in 3D Cartesian coordinates

        :return: projected coordinates using Azimuthal Equidistant Projection
        '''
        [r, elev, az] = self.cart2sph(pos[0], pos[1], pos[2])
        return self.pol2cart(az, m.pi / 2 - elev)

    def augment_EEG(self, data: np.ndarray, stdMult: float=0.1, pca: bool=False, n_components: int=2):
        '''
        Augment data by adding normal noise to each feature.

        Parameters:
        data            (np.ndarray) EEG feature data as a matrix 
                        (n_samples x n_features)
        stdMult         (float) Multiplier for std of added noise
        pca             (bool) if True will perform PCA on data and add noise proportional to PCA components.
        n_components    (int) Number of components to consider when using PCA.

        :return: Augmented data as a matrix (n_samples x n_features)
        '''
        augData = np.zeros(data.shape)
        if pca:
            pca = PCA(n_components=n_components)
            pca.fit(data)
            components = pca.components_
            variances = pca.explained_variance_ratio_
            coeffs = np.random.normal(scale=stdMult, size=pca.n_components) * variances
            for s, sample in enumerate(data):
                augData[s, :] = sample + (components * coeffs.reshape((n_components, -1))).sum(axis=0)
        else:
            # Add Gaussian noise with std determined by weighted std of each feature
            for f, feat in enumerate(data.transpose()):
                augData[:, f] = feat + np.random.normal(scale=stdMult*np.std(feat), size=feat.size)
        return augData

    def gen_images(self, locs: np.ndarray, feature_tensor: np.ndarray, n_gridpoints: int=32, normalize: bool=True, 
                augment: bool=False, pca: bool=False, std_mult: float=0.1, n_components: int=2, edgeless: bool=False):
        '''
        Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

        Parameters
        locs                (np.ndarray) An array with shape [n_electrodes, 2] containing X, Y
                            coordinates for each electrode.
        features            (np.ndarray) Feature matrix as [n_samples, n_features]
                            as [numchans, numfeatures, numsamples]
                            Features are as columns.
                            Features corresponding to each frequency band are concatenated.
                            (alpha1, alpha2, ..., beta1, beta2,...)
        n_gridpoints        (int) Number of pixels in the output images
        normalize           (bool) Flag for whether to normalize each band over all samples (default=True)
        augment             (bool) Flag for generating augmented images (default=False)
        pca                 (bool) Flag for PCA based data augmentation (default=False)
        std_mult            (float) Multiplier for std of added noise
        n_components        (int) Number of components in PCA to retain for augmentation
        edgeless            (bool) If True generates edgeless images by adding artificial channels
                            at four corners of the image with value = 0 (default=False).

        :return:            Tensor of size [samples, colors, W, H] containing generated
                            images.
        '''
        feat_array_temp = []            # list holder for feature array
        temp_interp = []

        numcontacts = feature_tensor.shape[0]     # Number of electrodes
        n_colors = feature_tensor.shape[1]
        n_colors = 4
        numsamples = feature_tensor.shape[2]    

        # Interpolate the values into a grid of x/y coords
        grid_x, grid_y = np.mgrid[min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                                 min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j]

        # loop through each color
        for c in range(n_colors):
            # build feature array from [ncontacts, 1 freq band, nsamples] squeezed and swapped axes
            feat_array_temp.append(feature_tensor[:,c,:].squeeze().swapaxes(0,1))
            
            # if c == 0:
                # print(feat_array_temp[0].shape)
            
            if augment: # add data augmentation -> either pca or not
                feat_array_temp[c] = self.augment_EEG(feat_array_temp[c], std_mult, pca=pca, n_components=n_components)

            # build temporary interpolator matrix    
            temp_interp.append(np.zeros([numsamples, n_gridpoints, n_gridpoints]))
        # Generate edgeless images -> add 4 locations (minx,miny),...,(maxx,maxy)
        if edgeless:
            min_x, min_y = np.min(locs, axis=0)
            max_x, max_y = np.max(locs, axis=0)
            locs = np.append(locs, np.array([[min_x, min_y], 
                                             [min_x, max_y],
                                             [max_x, min_y],
                                             [max_x, max_y]]), axis=0)
            for c in range(n_colors):
                feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((numsamples, 4)), axis=1)

       # Interpolating for all samples across all features
        for i in range(numsamples):
            for c in range(n_colors):
                temp_interp[c][i, :, :] = griddata(points=locs, 
                                            values=feat_array_temp[c][i, :], 
                                            xi=(grid_x, grid_y),
                                            method='cubic', 
                                            fill_value=np.nan)
            print('Interpolating {0}/{1}\r'.format(i+1, numsamples), end='\r')

        # Normalize every color (freq band) range of values
        for c in range(n_colors):
            if normalize:
                temp_interp[c][~np.isnan(temp_interp[c])] = scale(X = temp_interp[c][~np.isnan(temp_interp[c])])
            # convert all nans to 0
            temp_interp[c] = np.nan_to_num(temp_interp[c])

        # swap axes to have [samples, colors, W, H]
        return np.swapaxes(np.asarray(temp_interp), 0, 1)     

    def oldgen_images(self, locs: np.ndarray, feature_tensor: np.ndarray, n_gridpoints: int=32, normalize: bool=True, 
        augment: bool=False, pca: bool=False, std_mult: float=0.1, n_components: int=2, edgeless: bool=False):
        '''
        Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

        Parameters
        locs                (np.ndarray) An array with shape [n_electrodes, 2] containing X, Y
                            coordinates for each electrode.
        features            (np.ndarray) Feature matrix as [n_samples, n_features]
                            as [numchans, numfeatures, numsamples]
                            Features are as columns.
                            Features corresponding to each frequency band are concatenated.
                            (alpha1, alpha2, ..., beta1, beta2,...)
        n_gridpoints        (int) Number of pixels in the output images
        normalize           (bool) Flag for whether to normalize each band over all samples (default=True)
        augment             (bool) Flag for generating augmented images (default=False)
        pca                 (bool) Flag for PCA based data augmentation (default=False)
        std_mult            (float) Multiplier for std of added noise
        n_components        (int) Number of components in PCA to retain for augmentation
        edgeless            (bool) If True generates edgeless images by adding artificial channels
                            at four corners of the image with value = 0 (default=False).
        
        :return:            Tensor of size [samples, colors, W, H] containing generated
                            images.
        '''
        feat_array_temp = []            # list holder for feature array
        temp_interp = []

        numcontacts = locs.shape[0]     # Number of electrodes
        numsamples = features.shape[0]    

        # Test whether the feature vector length is divisible by number of electrodes
        assert features.shape[1] % numcontacts == 0
        # get the number of colors there are in image
        n_colors = features.shape[1] / numcontacts 

        # Interpolate the values into a grid of x/y coords
        grid_x, grid_y = np.mgrid[
                                 min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                                 min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                             ]

        # loop through each color
        for c in range(n_colors):
            # build feature array
            feat_array_temp.append(features[:, c * numcontacts : numcontacts * (c+1)])
            if augment: # add data augmentation -> either pca or not
                feat_array_temp[c] = self.augment_EEG(feat_array_temp[c], std_mult, pca=pca, n_components=n_components)

            # build temporary interpolator matrix    
            temp_interp.append(np.zeros([numsamples, n_gridpoints, n_gridpoints]))
        # Generate edgeless images -> add 4 locations (minx,miny),...,(maxx,maxy)
        if edgeless:
            min_x, min_y = np.min(locs, axis=0)
            max_x, max_y = np.max(locs, axis=0)
            locs = np.append(locs, np.array([[min_x, min_y], 
                                             [min_x, max_y],
                                             [max_x, min_y],
                                             [max_x, max_y]]), axis=0)
            for c in range(n_colors):
                feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((numsamples, 4)), axis=1)
       
       # Interpolating for all samples across all features
        for i in xrange(numsamples):
            for c in range(n_colors):
                temp_interp[c][i, :, :] = griddata(points=locs, 
                                            values=feat_array_temp[c][i, :], 
                                            xi=(grid_x, grid_y),
                                            method='cubic', 
                                            fill_value=np.nan)
            print('Interpolating {0}/{1}\r'.format(i+1, numsamples), end='\r')

        # Normalize every color (freq band) range of values
        for c in range(n_colors):
            if normalize:
                temp_interp[c][~np.isnan(temp_interp[c])] = scale(X = temp_interp[c][~np.isnan(temp_interp[c])])
            # convert all nans to 0
            temp_interp[c] = np.nan_to_num(temp_interp[c])

        # swap axes to have [samples, colors, W, H]
        return np.swapaxes(np.asarray(temp_interp), 0, 1)     