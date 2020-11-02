from costfunc import CostFunc
from deformations import create_gradient_map
from loaddata import LoadData
from entrypoint import EntryPoint
from distancemap import DistanceMap
from weights import dist_map_dict, LSB_class_group
from utils import binarize_segmentation
from coordinates import pad_seg_data
from glob import glob
import numpy as np
from os.path import join
import nrrd

pt_list = ['9']
wd = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort'

for pt in pt_list:
    # path to the loaded data
    img_pth = glob(join(wd, 'pt_{}\\*FFE*.nrrd'.format(pt)))[0]
    seg_pth = glob(join(wd, 'pt_{0}\\*{0}*egmentation-label.nrrd'.format(pt)))[0]
    lup_tbl = glob(join(wd, 'pt_{0}\\*{0}*egmentation-label_ColorTable.ctbl'.format(pt)))[0]
    fcsv = join(wd, 'pt_{}\\Craniotomy_Markers.fcsv'.format(pt))

    # outputted files

    MCF_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_{0}\\pt_{0}_MCF_ero3.csv'.format(pt)
    RS_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_{0}\\pt_{0}_RS_ero3.csv'.format(pt)


    seg_data, lut = LoadData.load_segmentation(seg_pth, lup_tbl)
    im, hdr = LoadData.load_image(img_pth)
    img_spacing = np.linalg.norm(hdr['space directions'],
                                 axis=1)  # slice spacing is defined as the normalized vector of the space transformation matrix

    # find entry points
    ep = EntryPoint(fcsv, hdr)
    MCF_entry, RS_entry = ep.from_fiducials()

    # pad the seg data
    seg_data = pad_seg_data(seg_data, MCF_entry)

    # save the padded image for visualization
    im_pad = pad_seg_data(im, MCF_entry)
    pad_pth = join(wd, 'pt_{0}\\pt_{0}_padded.nrrd'.format(pt))
    nrrd.write(pad_pth, im_pad, hdr)

    # create deformations
    gradient_maps = create_gradient_map(lut, seg_data, img_spacing)

    # create distance maps
    dist_maps = {}
    for name, subdict in dist_map_dict.items():
        group_memb = [key for key, value in LSB_class_group.items() if value["Group"] == subdict["Group"]]
        binarized_group = binarize_segmentation(seg_data, [lut[g] for g in group_memb])
        dm = DistanceMap(binarized_group, subdict["Max_Distance"], img_spacing)
        group_dist = dm.cdist("euclidean")
        dist_maps[name] = group_dist

    MCF_cf = CostFunc(MCF_entry, seg_data, lut, dist_maps, gradient_maps, img_spacing)
    MCF_cf.limit_cost()
    MCF_cf.save_dataframe(MCF_pth)

    RS_cf = CostFunc(RS_entry, seg_data, lut, dist_maps, gradient_maps, img_spacing)
    RS_cf.limit_cost()
    RS_cf.save_dataframe(RS_pth)
