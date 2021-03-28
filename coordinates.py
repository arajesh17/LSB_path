import numpy as np
from utils import unique2d, find_extrema, plot_point_cloud, find_center
from scipy.ndimage import generate_binary_structure, binary_erosion
from itertools import combinations

def create_crani_grid(parameters, offst, stepnum=5):
    """

    creates a grid connecting the vertices of the craniotomy grid

    Parameters
    ----------
    parameters: ndarray (4, 3)
        the coordinates for the vertices of the craniotomy
    offst: int
        the number of voxels from the border of grind you want to starte the grid coordinates. Important because if the
        point travels to the edge of the craniotomy techinically it will be outside of the bounds b/c of the radius of the cylinder
    stepnum: int
        the number of steps to take from corner points when creating the grid

    Returns
    -------

    output: ndarray (16, 3)
        array of grid coordinates of the craniotomy

    """

    origin = find_center(parameters)

    def sort_corners(param):
        """
        Find the pair of coordinates that have the longest distance, those will be defined as A and D, then find B and C
        # WARNING: This will fail with equilateral triangles
        :param param:
        :return:
        """

        combine = np.array(list(combinations(param, 2)))

        dist = 0
        for e in combine:
            d = np.linalg.norm(e[0] - e[1])
            if d > dist:
                pair = e
                dist = d

        _a = pair[0]
        _d = pair[1]

        # find which parts of param are not equal to a or d
        ad_loc = np.logical_or(np.all(param == _d, axis=1), np.all(param == _a, axis=1))
        bc_loc = [not x for x in ad_loc]
        _b, _c = param[bc_loc]

        return _a, _b, _c, _d

    a, b, c, d  = sort_corners(parameters)

    def create_steps(p1, p2, step_num=stepnum, offset=None):
        """ Create stepnum steps between p1 and p2 by calculating the unit vector between the two points and following the
        vector"""

        # create vector, step and unit vector (vhat)
        v = p2-p1
        step = np.linalg.norm(v)/step_num
        vhat = v/np.linalg.norm(v)

        if offset:
            # move p1 away from the boundary of crani by offset value
            p1_off = p1 + offset * vhat
            p1 = np.array([int(np.around(x)) for x in p1_off])

            p2_off = p2 - offset * vhat
            p2 = np.array([int(np.around(x)) for x in p2_off])

        pts = [p1]
        for j in range(1, step_num):
            # the new point is defined by i * the step * the unit vector (will create enough steps)
            p_new = p1 + j*step * vhat
            new_p = np.array([int(np.around(x)) for x in p_new])
            pts.append(new_p)
        pts.append(p2)
        return pts

    # create the steps for all the edges of the square (ab, ac, bd, cd)
    ab = create_steps(a, b, offset=offst)
    ac = create_steps(a, c, offset=offst)
    bd = create_steps(b, d, offset=offst)
    cd = create_steps(c, d, offset=offst)

    output = ab
    # go down the line from ac and bd and create steps between ac[i] and bd[i] for i in range(stepnum)
    for i in range(1, stepnum):
        arr = create_steps(ac[i], bd[i])
        output = np.vstack((output, arr))
    output = np.vstack((output, cd))
    output = unique2d(output)

    return output



def find_edge_points(target, struct, iter=1):
    """
    Finds the boundary points of the target by eroding it by three iterations
    Then it finds the extrema of x,y,z points

    :param target: binary mask of the target segmentation
    :param struct: structure of the kernel to apply for erosion
    :param iter: int number of iterations to erode
    :return:
    """

    """
    # erode 1 time but with a scaled structure reflecting the diameter of the cylinders that will be access channels
    for i in range(iter):
        eroded = binary_erosion(target, structure=struct, iterations=iter).astype(int)
        if len(np.where(eroded != 0)[0]) == 0: # make sure you don't iterate so many times it shirnks the target to nothing
            break
        target = eroded
    """

    # key points:
    key_points = []
    coords = np.array(np.where(target == 1)).T

    low = find_extrema(coords, 'min')
    high = find_extrema(coords, 'max')

    for d in [[0, 1, 2], [2, 0, 1], [1, 2, 0]]:

        # find low
        low_dim1 = coords[coords[:, d[0]] == low[d[0]]]

        max_dim2 = low_dim1[low_dim1[:, d[1]] == np.max(low_dim1[:, d[1]])]
        max_dim3 = max_dim2[max_dim2[:, d[2]] == np.max(max_dim2[:, d[2]])][0]
        key_points.append(max_dim3)
        min_dim3 = max_dim2[max_dim2[:, d[2]] == np.min(max_dim2[:, d[2]])][0]
        key_points.append(min_dim3)

        min_dim2 = low_dim1[low_dim1[:, d[1]] == np.min(low_dim1[:, d[1]])]
        max_dim3 = min_dim2[min_dim2[:, d[2]] == np.max(min_dim2[:, d[2]])][0]
        key_points.append(max_dim3)
        min_dim3 = min_dim2[min_dim2[:, d[2]] == np.min(min_dim2[:, d[2]])][0]
        key_points.append(min_dim3)

        # find high
        high_dim1 = coords[coords[:, d[0]] == high[d[0]]]

        max_dim2 = high_dim1[high_dim1[:, d[1]] == np.max(high_dim1[:, d[1]])]
        max_dim3 = max_dim2[max_dim2[:, d[2]] == np.max(max_dim2[:, d[2]])][0]
        key_points.append(max_dim3)
        min_dim3 = max_dim2[max_dim2[:, d[2]] == np.min(max_dim2[:, d[2]])][0]
        key_points.append(min_dim3)

        min_dim2 = high_dim1[high_dim1[:, d[1]] == np.min(high_dim1[:, d[1]])]
        max_dim3 = min_dim2[min_dim2[:, d[2]] == np.max(min_dim2[:, d[2]])][0]
        key_points.append(max_dim3)
        min_dim3 = min_dim2[min_dim2[:, d[2]] == np.min(min_dim2[:, d[2]])][0]
        key_points.append(min_dim3)

    plot = False
    if plot:
        pts = np.vstack(tuple(x for x in key_points))
        plot_point_cloud(np.where(target == 1), pts.T)

    key_points = unique2d(np.array(key_points))

    return key_points[::1]

def convert_ijk_to_RAS(hdr, pt):
    """
    converts points from RAS to ijk

    Parameters
    ----------
    hdr: nrrd header
        the header of image
    pt: ndarray (3,) or (n,3)
        array with points in RAS space to transform to ijk

    Returns
    -------

    output: ndarray
        points transformed into the ijk space

    """

    # turn into the [x,y,z, 1] format
    if pt.shape == (3,):
        pt = np.concatenate((pt, [1])).reshape(-1, 1)

    if pt.shape != (4,1):
        return ValueError('Point shape is {} but needs to be (4,1))'.format(pt.shape))

    spacing = np.hstack((hdr['space directions'].T, hdr['space origin'].reshape(-1, 1)))
    ijk_to_lps = np.vstack((spacing, [0, 0, 0, 1]))
    lps_to_ras = np.diag([-1, -1, 1, 1])
    ijk_to_ras = np.matmul(lps_to_ras, ijk_to_lps)
    x_pt = np.matmul(ijk_to_ras, pt)
    return x_pt.flatten()[:-1]


def convert_RAS_to_ijk(hdr, pt):
    """
    converts points from RAS to ijk

    Parameters
    ----------
    hdr: nrrd header
        the header of image
    pt: ndarray (3,) or (n,3)
        array with points in RAS space to transform to ijk

    Returns
    -------

    output: ndarray
        points transformed into the ijk space

    """

    # turn into the [x,y,z, 1] format
    if pt.shape == (3,):
        pt = np.concatenate((pt, [1])).reshape(-1,1)

    if pt.shape != (4, 1):
        return ValueError('Point shape is {} but needs to be (4,1))'.format(pt.shape))

    spacing = np.hstack((hdr['space directions'].T, hdr['space origin'].reshape(-1, 1)))
    ijk_to_lps = np.vstack((spacing, [0, 0, 0, 1]))
    lps_to_ras = np.diag([-1, -1, 1, 1])
    ijk_to_ras = np.matmul(lps_to_ras, ijk_to_lps)
    ras_to_ijk = np.linalg.inv(ijk_to_ras)
    x_pt = np.round(np.matmul(ras_to_ijk, pt))

    return x_pt.flatten()[:-1]

def pad_seg_data(seg_data, crani_coords, ax=2):
    """
    Pads the segmentation data in the specific axis dimension

    :param seg_data: ndarray (n_voxel_X, n_voxel_Y, n_voxel_Z)
                The voxelized data of the segmentation
    :param  crani_coords: ndarray (n, 3)
                the coordinates of the craniotomy points from the fcsv file
    :return:

    """

    # check if the crani coords are within the geometry of the seg_data
    upper = 0
    lower = 0
    for pts in crani_coords:
        if np.max(crani_coords[:, ax]) > upper:
            upper = np.max(crani_coords[:, ax])
        if np.min(crani_coords[:, ax]) < lower:
            lower = np.min(crani_coords[:, ax])

    upper_pad = 0
    if upper >= seg_data.shape[ax]:
        upper_pad = upper + 1 - seg_data.shape[ax]

    pad_im = np.repeat(seg_data, [1]*(seg_data.shape[ax]-1) + [upper_pad], axis=ax)

    return pad_im
