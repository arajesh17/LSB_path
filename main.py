from scipy import optimize
from costfunction import CostFunction
from costfunc import CostFunc
from deformations import create_gradient_map
from loaddata import LoadData
from entrypoint import EntryPoint
from bounds import RandomDisplacementBounds
from distancemap import DistanceMap
from weights import dist_map_dict, get_group_members, LSB_class_weights, LSB_class_group
from utils import binarize_segmentation, find_extrema, plot_path_cost, pad_seg_data
import numpy as np
from save import save_path

# path to the loaded data
img_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\601 AX 3D B FFE IACs_1.nrrd'
seg_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\Pt_6_Segmentation-label.nrrd'
lup_tbl = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\Pt_6_Segmentation-label_ColorTable.ctbl'
fcsv = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\F.fcsv'

seg_data, lut = LoadData.load_segmentation(seg_pth, lup_tbl)
ct, hdr = LoadData.load_image(img_pth)
img_spacing = np.linalg.norm(hdr['space directions'],
                             axis=1)  # slice spacing is defined as the normalized vector of the space transformation matrix

# create distance maps
dist_maps = {}
for name, subdict in dist_map_dict.items():
    group_memb = [key for key, value in LSB_class_group.items() if value["Group"] == subdict["Group"]]
    binarized_group = binarize_segmentation(seg_data, [lut[g] for g in group_memb])
    dm = DistanceMap(binarized_group, subdict["Max_Distance"], img_spacing)
    group_dist = dm.cdist("euclidean")
    dist_maps[name] = group_dist

# create deformations
gradient_maps = create_gradient_map(lut, seg_data, img_spacing)

# find entry points
ep = EntryPoint(fcsv, hdr)
MCF_entry, RS_entry = ep.from_fiducials()

#entry = np.array([[398, 378, 57], [363, 474, 52], [331, 420, 1], [381, 367, 2]])
MCF_cf = CostFunc(MCF_entry, seg_data, lut, dist_maps, gradient_maps, img_spacing)
MCF_cf.limit_cost() # What does this return?, how to save the files in pickl

RS_cf = CostFunc(RS_entry, seg_data, lut, dist_maps, gradient_maps, img_spacing)
RS_cf.limit_cost()



