from utils import binarize_segmentation
from scipy.ndimage import binary_erosion
from weights import LSB_class_group
import numpy as np

# 1 get the lut with the different structures
# 2 find the group which they belong to
# 3 then create deformability map for them


def create_gradient_map(lut, data, img_spacing, group_table= LSB_class_group):
    """ Create gradient map"""

    gradient_maps = {}
    def_group = [k for k, v in group_table.items() if v["Class"] == "Deformable"]
    for group in def_group:
        def_struct = binarize_segmentation(data, [lut[s] for s in lut.keys() if group_table[s]["Group"] == group])
        n_iter = group_table[group]['Iterations']
        def_grad = create_erosion_gradient(def_struct, img_spacing, n_iter)
        gradient_maps[group] = def_grad

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
