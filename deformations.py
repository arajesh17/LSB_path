from utils import binarize_segmentation
from scipy.ndimage import binary_erosion
from weights import LSB_class_group
import numpy as np
import nrrd


# 1 get the lut with the different structures
# 2 find the group which they belong to
# 3 then create deformability map for them


def create_gradient_map(lut, data, img_spacing, group_table= LSB_class_group):
    """
    Create the gradient map of the specified structures by using  binary erosion


    Parameters
    ----------
    lut: dictionary
        look up table encoded by the color table which pairs integers with semantic segmentation strings
    data: ndarray (n_voxel_X, n_voxel_Y, n_voxel_Z)
        segmentation data
    img_spacing: ndarray (1,3)
        voxel spacing of data
    group_table: dictionary
        the table of the group properties

    Returns
    -------

    gradient maps: dictionary {"group": group_gradient}
        dictionary with the respective eroded gradient map as the value to the group key

    """

    gradient_maps = {}
    def_group = [k for k, v in group_table.items() if v["Class"] == "Deformable"]
    for group in def_group:
        def_struct = binarize_segmentation(data, [lut[s] for s in lut.keys() if group_table[s]["Group"] == group])
        n_iter = group_table[group]['Iterations']
        def_grad = create_erosion_gradient(def_struct, img_spacing, n_iter)
        def_grad_scaled = set_erosion_gradient(def_grad, group_table[group]['Weight'], group_table[group]['Factor'])
        gradient_maps[group] = def_grad_scaled

    return gradient_maps

def create_erosion_gradient(data, spacing, n_iter, runs=5):
    """ Creates an erosiion gradient for the images used for the deformability approximation"""

    step_dim = 10 # dimensions of each step in mm

    struct = np.ones((int(np.rint(step_dim/spacing[0])),
                      int(np.rint(step_dim/spacing[1])),
                      int(np.rint(step_dim/spacing[2]))))

    gradient_map = np.zeros(data.shape)
    for i in np.arange(1, runs+1):
        step = binary_erosion(data, structure=struct, iterations=n_iter).astype(int)
        gradient_map[np.where((data - step) == 1)] = i
        data = step

    return gradient_map

def set_erosion_gradient(data, base, factor):
    """
    allows for scaling of gradient_map based on input weights

    Parameters
    ----------
    data: ndarray gradient mask
    base: base scalar to set all elements of the gradient
    factor: factor scalar to increase at each gradient

    Returns
    -------
    gradient_map_scaled: ndarray of gradient map that is scaled to the appropriate values

    """
    factor_map = data * factor
    base_map = np.zeros(data.shape)
    base_map[np.where(data > 0)] = base - factor
    gradient_map_scaled = base_map + factor_map

    return gradient_map_scaled

def save_gradient_maps(outpth, hdr, gradientmaps):
    """
    Saves the gradient maps in one .nrrd file to use at output

    Parameters
    ----------
    outpth: basestring
        The output string where you want to save the gradient maps
    gradientmaps: dict
        The dictionary of the gradient maps created by create_gradient_maps method
    hdr: dict
        The hdr used to save the nrrd file output

    Returns
    -------
    n/a

    """

    combined_grad_map = np.sum(list(gradientmaps.values()), axis=0)
    print('Writing combined gradient path to {}'.format(outpth))
    nrrd.write(outpth, combined_grad_map, hdr)
