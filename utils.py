import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from weights import microscope_dict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return np.int(obj)
        return json.JSONEncoder.default(self, obj)

def load_json(file):

    with open(file, 'r') as fp:
        d = json.load(fp)
    return d


def find_center(coordinates):
    """
    finds the center of mass of coordinates in 3 dimensions

    Parameters
    ----------
    coordinates: ndarray (n,3)
        array of coordinates to find the center of

    Returns
    -------
    com: ndarray (1,3)
        the center of mass of the coordinates

    """

    if len(coordinates.shape) == 3:
        coordinates = np.array(np.where(coordinates > 0)).T

    if coordinates.shape == (3,):
        return coordinates

    com = np.array([np.sum(coordinates[:, 0]) / len(coordinates[:, 0]),
           np.sum(coordinates[:, 1]) / len(coordinates[:, 1]),
           np.sum(coordinates[:, 2]) / len(coordinates[:, 2])])

    return com


def find_extrema(coordinates, minmax):
    """

    :param coordinates: np array with shape (n, n2) with format [[x1, y1, z1], [x2, y2, z2], ....]
    :param minmax: 'string' either 'min' or 'max'
    :return:
          extrema: n2 sized array with outputs from extrema in each of the dimensions of n2
    """
    if len(coordinates.shape) == 1:
        return coordinates

    if minmax == 'min':
        extrema = [np.min(coordinates[:, i]) for i in range(coordinates.shape[-1])]
    elif minmax == 'max':
        extrema = [np.max(coordinates[:, i]) for i in range(coordinates.shape[-1])]

    return extrema


def create_circle(r1, r2, n=20):
    """ creates a circle of points with radius and n for number of coords"""

    t = np.linspace(0, 2 * np.pi, num=n)
    cos = r1 * np.cos(t)
    sin = r2 * np.sin(t)
    return cos, sin


def get_angle(v1, v2):
    """
    finds the angle between two vectors in radians
    :param v1: vector 1
    :param v2: vector 2
    :return:
    """

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    angle_of_rot = np.arccos(np.dot(v1, v2))

    return angle_of_rot

def create_rotation_matrix(vector1, vector2):
    """
    to generate the rotation matrix between two vectors we calculate
    the cross product which creates the orthogonal vector to the two vectors
    then we rotate by the angle defined by the dot product around the axis
    to create our rotation matrix which can be applied to any coordinates

    Parameters
    ----------
    vector 1: array
        vector1 ex. [100, 0 , 0]
    vector 2: array
        vector2  ex. [0, 100, 0]

    Returns
    -------
    rotation: scipy.spatial.transform.rotation object
         see docs https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

    """

    axis1_v_hat = vector1 / np.linalg.norm(vector1)
    axis2_v_hat = vector2 / np.linalg.norm(vector2)

    axis_of_rot = np.cross(axis1_v_hat, axis2_v_hat)
    if np.linalg.norm(axis_of_rot) != 0: axis_of_rot = axis_of_rot / np.linalg.norm(axis_of_rot)
    angle_of_rot = np.arccos(np.dot(axis1_v_hat, axis2_v_hat))

    rotation = R.from_rotvec(axis_of_rot*angle_of_rot)

    return rotation


def plot_convhull(hull, pts = [], axes=[], plt_title=''):
    """ visualizes convex hull with faces all drawn"""


    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as colors
    import scipy as sp

    _pts = hull.points

    fig = plt.figure()
    ax = Axes3D(fig)
    if len(axes) > 0:
        ax.set_xlim3d(axes[0])
        ax.set_ylim3d(axes[1])
        ax.set_zlim3d(axes[2])
    else:
        ax.set_xlim3d(np.min(_pts[:, 0]), np.max(_pts[:, 0]))
        ax.set_ylim3d(np.min(_pts[:, 1]), np.max(_pts[:, 1]))
        ax.set_zlim3d(np.min(_pts[:, 2]), np.max(_pts[:, 2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(plt_title)

    faces = hull.simplices

    for s in faces:

        sq = [
            [_pts[s[0], 0], _pts[s[0], 1], _pts[s[0], 2]],
            [_pts[s[1], 0], _pts[s[1], 1], _pts[s[1], 2]],
            [_pts[s[2], 0], _pts[s[2], 1], _pts[s[2], 2]]
        ]

        f = Poly3DCollection([sq])
        f.set_color(colors.rgb2hex(sp.rand(3)))
        f.set_edgecolor('k')
        f.set_alpha(0.5)
        ax.add_collection3d(f)

#    if pts:
#        ax.plot(pts[0], pts[1], pts[2], 'ro')

    plt.show(block=False)

def plot_path_cost(dict_list):

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    viridis = mpl.cm.get_cmap('viridis', 12)

    def norm(val):
        return 1000000 - val / 1000000


    for costfunc, cost in dict_list.items():

       ep = costfunc.entry_point
       targ = find_center(costfunc.target_coords)
       line = np.vstack((ep, targ)).T
       ax.plot(line[0], line[1], line[2],
               label='{} {}'.format(costfunc.entry_point, cost), color=viridis(norm(cost)), linestyle='-', marker='o')

    plt.legend()
    plt.show()


def plot_heat_map(targ, crani, min_dist, title, bounds=[0,30]):
    """
    Allows for visualization fo the paths with heat map included for coolness factor

    :param targ:
    :param crani:
    :param min_dist:
    :param title:
    :param bnds:
    :return:
    """

    ax = Axes3D(plt.figure())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    step = np.arange(bounds[0], bounds[1], int((bounds[1] - bounds[0]) / 5), dtype=int)
    norm = mpl.colors.Normalize(vmin=step.min(), vmax=step.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet_r)
    cmap.set_array([])

    for idx in range(len(min_dist)):

        c = crani[idx]
        pts = np.vstack((c, targ)).T
        ax.plot(pts[0], pts[1], pts[2], marker='o', linestyle='-',
                    color=cmap.to_rgba(min_dist[idx]), label=c)


    # adjust size of colorbar
    axins = inset_axes(ax,
                       width="5%",
                       height="80%",
                       borderpad=2) # much more u can add with this function

    plt.colorbar(cmap, cax=axins, ticks=step)

    plt.show()


def plot_point_cloud(pts, pts2=[], axes=[]):
    """ plt 3d collection of points"""

    ax = Axes3D(plt.figure())
    ax.plot(pts[0], pts[1], pts[2], 'ro')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if len(axes) > 0:
        ax.set_xlim3d(axes[0])
        ax.set_ylim3d(axes[1])
        ax.set_zlim3d(axes[2])

    if len(pts2):
        ax2 = Axes3D(plt.figure())
        ax2.plot(pts2[0], pts2[1], pts2[2], 'bo')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

    plt.show(block=False)

def binarize_segmentation(input_segmentation, values):
    """
    Binarize Segmentations takes the input segmentation of a labeled masks and creates an individually binarized masks

    Parameters
    --------
    input_segmentation : nparray
        segmented labeled binary map. 3D array with voxel values corresponding to tissue type
    values :  list of ints
        list of integer value of the label from the input_segmentation which we want to binarize

    Returns
    --------
    binarized_seg: nparray
        binarized file of the segmentation
    """
    binarized_seg = np.zeros(input_segmentation.shape)
    binarized_seg[np.isin(input_segmentation, values)] = 1
    return binarized_seg

def array_to_coords(myarray):
    """ converts a voxel array into a (n,3) array with all non-zero coordinates"""
    non_zero = np.where(myarray > 0 )
    points = np.array(list(zip(non_zero[0], non_zero[1], non_zero[2])))
    return points

def unique2d(arr,consider_sort=False,return_index=False,return_inverse=False):
    """Get unique values along an axis for 2D arrays.
        Ripped from stack overflow: https://stackoverflow.com/questions/41234161/check-common-elements-of-two-2d-numpy-arrays-either-row-or-column-wise

        input:
            arr:
                2D array
            consider_sort:
                Does permutation of the values within the axis matter?
                Two rows can contain the same values but with
                different arrangements. If consider_sort
                is True then those rows would be considered equal
            return_index:
                Similar to numpy unique
            return_inverse:
                Similar to numpy unique
        returns:
            2D array of unique rows
            If return_index is True also returns indices
            If return_inverse is True also returns the inverse array
            """

    if consider_sort is True:
        a = np.sort(arr,axis=1)
    else:
        a = arr
    b = np.ascontiguousarray(a).view(np.dtype((np.void,
            a.dtype.itemsize * a.shape[1])))

    if return_inverse is False:
        _, idx = np.unique(b, return_index=True)
    else:
        _, idx, inv = np.unique(b, return_index=True, return_inverse=True)

    if return_index == False and return_inverse == False:
        return arr[idx]
    elif return_index == True and return_inverse == False:
        return arr[idx], idx
    elif return_index == False and return_inverse == True:
        return arr[idx], inv
    else:
        return arr[idx], idx, inv


def in2d_unsorted(arr1, arr2, axis=1, consider_sort=False):
    """Find the elements in arr1 which are also in
       arr2 and sort them as the appear in arr2

    https://stackoverflow.com/questions/41234161/check-common-elements-of-two-2d-numpy-arrays-either-row-or-column-wise

    """

    assert arr1.dtype == arr2.dtype

    if axis == 0:
        arr1 = np.copy(arr1.T,order='C')
        arr2 = np.copy(arr2.T,order='C')

    if consider_sort is True:
        sorter_arr1 = np.argsort(arr1)
        arr1 = arr1[np.arange(arr1.shape[0])[:,None],sorter_arr1]
        sorter_arr2 = np.argsort(arr2)
        arr2 = arr2[np.arange(arr2.shape[0])[:,None],sorter_arr2]


    arr = np.vstack((arr1,arr2))

    _, inv = unique2d(arr, return_inverse=True)

    size1 = arr1.shape[0]
    size2 = arr2.shape[0]

    arr3 = inv[:size1]
    arr4 = inv[-size2:]

    # Sort the indices as they appear in arr2
    sorter = np.argsort(arr3)
    idx = sorter[arr3.searchsorted(arr4, sorter=sorter)]

    return idx

def reflect(im, midpoint, ax=2):
    """

    Parameters
    ----------
    im: array
        array of segmented image
    midpoint: int
        midsaggital plane along which the segmentation will be reflected
    axis: int
        axis along which the array will be reflected

    Returns
    -------
    reflected_im: array
        array with the segmented image reflected over the midsagital plane
    """

    if ax != 2:
        return ValueError("Code not set up for axis other than 2, please add into method")

    size = im.shape[2]
    rev_start = midpoint - (size-midpoint)

    normal_im = im[:, :, midpoint:]
    rev_im = np.flip(normal_im, axis=ax)

    reflected_im = np.zeros(im.shape)
    reflected_im[:, :, midpoint:] = normal_im
    reflected_im[:, :, rev_start:midpoint] = rev_im

    return reflected_im

def normalize_head_size(dist_pt, seg_data, lut, img_spacing):
    """
    method to find the head size to normalize raw costs based on patient head size. The distal point is point furthest
    from center of the brain
    Parameters
    ----------
    dist_pt: ndarray
        3d coordinates of the distal point to measure
    seg_data: ndarray
        segmentation of structures
    lut: dict
        dictionary of names of structures with corresponding integer values
    img_spacing:
        voxel dimension of the image to normalize the size to mm
    Returns
    -------

    """

    # find center of mass of the cochlea segmentation
    cochlea = binarize_segmentation(seg_data, lut['Cochlea'])
    prox_pt = center_of_mass(cochlea)

    # distance between prox_pt and dist_pt normalized by img_spacing
    head_size = np.sqrt(np.sum(((prox_pt-dist_pt) * img_spacing)**2))

    return head_size