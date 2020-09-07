from scipy import optimize
from costfunction import CostFunction
from loaddata import LoadData
from entrypoint import EntryPoint
from bounds import CraniBounds
from weights import dist_map_dict, get_group_members, LSB_class_weights, LSB_class_group, microscope_dict
from surgicalpath import SurgicalPath
from utils import binarize_segmentation
from deformations import create_gradient_map
import numpy as np
import nrrd
from costfunction import array_to_coords

## path to the loaded data
#img_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\601 AX 3D B FFE IACs_1.nrrd'
#seg_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\Pt_6_Segmentation-label.nrrd'
#
#lup_tbl = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\Pt_6_Segmentation-label_ColorTable.ctbl'
#
#seg_data, lut = LoadData.load_segmentation(seg_pth, lup_tbl)
#ct, hdr = LoadData.load_image(img_pth)
#img_spacing = np.linalg.norm(hdr['space directions'],
#                             axis=1)  # slice spacing is defined as the normalized vector of the space transformation matrix
#
## create distance maps
#dist_maps = {}
#for name, subdict in dist_map_dict.items():
#    group_memb = [key for key, value in LSB_class_group.items() if value["Group"] == subdict["Group"]]
#    binarized_group = binarize_segmentation(seg_data, [lut[g] for g in group_memb])
#    group_dist = LoadData.create_distance_map(binarized_group, subdict["Max_Distance"])
#    dist_maps[name] = group_dist
#
## create deformations
#gradient_maps = create_gradient_map(lut, seg_data, img_spacing)
#
## find entry points
#RS_ep = EntryPoint(binarize_segmentation(seg_data, lut["RS"]),
#                   binarize_segmentation(seg_data, lut["Superficial_Tissue"]))
#RS_ep.skull_strip()
#RS_coords = RS_ep.get_entry_points()
#
#
#ep = np.array([375, 444, 43])
#target_struct = binarize_segmentation(seg_data, lut["Target"])
#target_coords = array_to_coords(target_struct)
#
#
#initial_guess = np.array([413, 363, 25])
#result = CostFunction.costfunct_dictionary(initial_guess, RS_coords, seg_data, lut,
#                                           dist_maps, gradient_maps, img_spacing)

from glob import glob
from os.path import join
from loaddata import LoadData
import numpy as np
from coordinates import find_edge_points

pt_list = ['9']
wd = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort'

for pt in pt_list:
    # path to the loaded data
    img_pth = glob(join(wd, 'pt_{}\\*FFE*.nrrd'.format(pt)))[0]
    seg_pth = glob(join(wd, 'pt_{0}\\*{0}*egmentation-label.nrrd'.format(pt)))[0]
    lup_tbl = glob(join(wd, 'pt_{0}\\*{0}*egmentation-label_ColorTable.ctbl'.format(pt)))[0]
    fcsv = join(wd, 'pt_{}\\Craniotomy_Markers.fcsv'.format(pt))

    # outputted files

    MCF_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_{0}\\pt_{0}_MCF.csv'.format(pt)
    RS_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_{0}\\pt_{0}_RS.csv'.format(pt)


    seg_data, lut = LoadData.load_segmentation(seg_pth, lup_tbl)
    im, hdr = LoadData.load_image(img_pth)
    img_spacing = np.linalg.norm(hdr['space directions'],
                                 axis=1)  # slice spacing is defined as the normalized vector of the space transformation matrix

    # get edge points
    out_im = np.zeros(im.shape)
    targ = binarize_segmentation(seg_data, lut['Target'])
    struct = np.ones((int(np.rint(1 / img_spacing[0])),
                      int(np.rint(1 / img_spacing[1])),
                      int(np.rint(1 / img_spacing[2]))))
    edge = find_edge_points(targ, struct)

    for ky in edge:
        out_im[ky[0], ky[1], ky[2]] = 1

    # save the new targets
    out_pth = join(wd, 'pt_{}\\target_scaled_iter.nrrd'.format(pt))
    nrrd.write(out_pth, out_im, hdr)


