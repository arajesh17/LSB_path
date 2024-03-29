from costfunc import CostFunc
from deformations import create_gradient_map, save_gradient_maps
from loaddata import LoadData
from entrypoint import EntryPoint
from distancemap import DistanceMap
from weights import dist_map_dict, LSB_class_group
from utils import binarize_segmentation, normalize_head_size
from coordinates import pad_seg_data
from glob import glob
import numpy as np
from os.path import join
import nrrd

pt_list = ['27']
wd = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort'
image_extension = 'T2_1x1x1'
#image_extension = 'FFE'

for pt in pt_list:

    # path to the loaded data
    img_pth = glob(join(wd, 'pt_{}\\*{}*.nrrd'.format(pt, image_extension)))[0]
    seg_pth = glob(join(wd, 'pt_{0}\\*{0}*egmentation-label_1x1x1.nrrd'.format(pt)))[0]
    lup_tbl = glob(join(wd, 'pt_{0}\\*{0}*egmentation-label_1x1x1_ColorTable.ctbl'.format(pt)))[0]
    fcsv = join(wd, 'pt_{}\\Craniotomy_Markers.fcsv'.format(pt))

    # outputted files
    MCF_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_{0}\\pt_{0}_MCF_1x1x1_weights.csv'.format(pt)
    RS_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_{0}\\pt_{0}_RS_1x1x1_weights.csv'.format(pt)


    # load the data space it correctly
    seg_data, lut = LoadData.load_segmentation(seg_pth, lup_tbl)
    im, hdr = LoadData.load_image(img_pth)
    img_spacing = np.linalg.norm(hdr['space directions'],
                                 axis=1)  # slice spacing is defined as the normalized vector of the space transformation matrix

    # find entry points
    ep = EntryPoint(fcsv, hdr)
    MCF_entry, RS_entry = ep.from_fiducials()

    # normalize with the head size #TODO move to cost function
    head_size = normalize_head_size(ep.sigmoid, seg_data, lut, img_spacing)

# TODO resample images to uniform resolution

    # pad the seg data
    seg_data = pad_seg_data(seg_data, MCF_entry)

    # save the padded image for visualization
    im_pad = pad_seg_data(im, MCF_entry)
    pad_pth = join(wd, 'pt_{0}\\pt_{0}_padded.nrrd'.format(pt))
    nrrd.write(pad_pth, im_pad, hdr)

    # create deformations
    gradient_maps = create_gradient_map(lut, seg_data, img_spacing)
    gradient_map_pth = join(wd, 'pt_{0}\\pt_{0}_gradient.nrrd'.format(pt))
    save_gradient_maps(gradient_map_pth, hdr, gradient_maps)


    # create distance maps currently not in the code but can be added if distance is a parameter
    dm = DistanceMap(seg_data, '5' , img_spacing)
    dm.get_distmaps()

    MCF_cf = CostFunc(MCF_entry, seg_data, lut, dm.distmaps, gradient_maps, img_spacing)
    MCF_cf.limit_cost()
    MCF_cf.save_dataframe(MCF_pth)

    RS_cf = CostFunc(RS_entry, seg_data, lut, dm.distmaps, gradient_maps, img_spacing)
    RS_cf.limit_cost()
    RS_cf.save_dataframe(RS_pth)
