from scipy.spatial import KDTree, distance
import numpy as np
from time import time
from loaddata import LoadData
from utils import binarize_segmentation, in2d_unsorted, unique2d


class DistanceMap(object):

    def __init__(self, seg_data, max_dist, img_spacing):

        seg_coords = np.array(np.where(seg_data == 1)).T

        # calibrated to the actual mm size of the image
        lowlim = np.array([np.int64(np.min(seg_coords[:, x]) - max_dist/img_spacing[x]) for x in [0,1,2]])
        lowlim = lowlim * np.array(lowlim >= 0)

        # calibrated to the actual mm size of the image
        highlim = np.array([np.int64(np.max(seg_coords[:, x]) + max_dist/img_spacing[x]) for x in [0,1,2]])
        for i, val in enumerate(highlim):
            if val >= seg_data.shape[i]:
                highlim[i] = np.int64(seg_data.shape[i]-1)


        self.lowlim = lowlim
        self.highlim = highlim
        self.segdata = seg_data
        self.segcoords = seg_coords
        self.maxdist = max_dist
        self.imspacing = img_spacing


    def cdist(self, method):
        """

        find distance between points using the cdist method

        Parameters
        ----------
        method: string ["euclidian", "wminkowski"]
            the method for calculating distance
        Returns
        -------
        distance_map: ndarray [n_voxel_X, n_voxel_Y, n_voxel_Z]
            the distance between points in the voxel space

        """
        """
        
        find distance using cdist approach
        :param method (str) 'euclidian' or 'wminkowski'
        :return:
        """

        if method not in ['euclidean', 'wminkowski']:
            raise ValueError("method passed was incorrect")

        im = np.zeros(self.segdata.shape)
        xx, yy, zz = np.meshgrid(np.arange(self.lowlim[0], self.highlim[0] + 1),
                                 np.arange(self.lowlim[1], self.highlim[1] + 1),
                                 np.arange(self.lowlim[2], self.highlim[2] + 1))


        testpoints = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        # remove the points from self.seg coords from the testpoints array
        valid = in2d_unsorted(testpoints, self.segcoords)
        bool_valid = np.ones(testpoints.shape[0])
        bool_valid[valid] = 0
        testpoints = testpoints[bool_valid.astype(bool)]

        for p in testpoints:
            p = p.reshape(1, 3).astype(int)

            if method == 'euclidean':
                min_dist = np.min(distance.cdist(p, self.segcoords,
                                                 'euclidean'))
            # winkowski allows for scaled distances based on actual voxel dimensions so that it is accurate in terms of physical distance
            elif method == 'winkowski':
                min_dist = np.min(distance.cdist(p, self.segcoords,
                                                 'wminkowski',
                                                 p=2,
                                                 w=self.imspacing))

            im[p[0,0], p[0,1], p[0,2]] = min_dist

        # turn all other values to max_distance
        maxval = np.max(np.unique(im))
        im[np.where(im == 0)] = maxval

        # turn the coordinates of the structure to zero
        im[self.segcoords.T[0], self.segcoords.T[1], self.segcoords.T[2]] = 0

        return im
