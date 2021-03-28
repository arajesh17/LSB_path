from glob import glob
import matplotlib as mpl
import nrrd
from os.path import join
import pandas as pd
import numpy as np
from surgicalpath import Cylinder
from scipy.ndimage import binary_dilation
from weights import limit_dict
from utils import load_json
import re

def correct_1d_array(input, type=np.int):
    """ correct the 2d arrays that were saved in the csv"""
    arr = np.array([float(x) for x in re.findall(r"\d+(?:\.\d+)?|\b(?:inf)", input)])
    arr = arr.astype(type)
    return arr

def correct_numpy_array(input, dim=3, t=np.int):

    #first get all the number characters
    arr = np.array([float(x) for x in re.findall(r"\d+(?:\.\d+)?|\b(?:inf)", input)])
    arr = arr.astype(t)
    arr = arr.reshape(int(len(arr)/dim), dim)

    return arr


def correct_2d_array(input, type=int):
    """
    :param input: correct the 2d arrays that were ssave in th csv
    :return:
    """
    arr = np.empty((0, 3), dtype=type)
    for row in input.split('\n'):
        arr = np.vstack((arr, correct_1d_array(row, type=type)))
    return arr

def correct_df(df):
    """
    reformats dataframe from string in cells to numpy arrays
    :param df:
    :return:
    """

    df['crani'] = df['crani'].apply(correct_numpy_array)
    df['cost'] = df['cost'].apply(correct_1d_array, type=np.float)
    df['target'] = df['target'].apply(correct_1d_array)
    for col in df.columns.values[4:]:
        df[col] = df[col].apply(correct_numpy_array, dim=2, t=float)

    return df

def create_seg_hdr(im_hdr, seg_hdr, seg_data, bnds):
    """
    attempt to make the
    :param im_hdr:
    :param seg_hdr:
    :param seg_data:
    :return:
    """

    hdr = {}
    hdr['type'] = 'unsigned char'
    hdr['dimension'] = 4
    hdr['space'] = im_hdr['space']
    hdr['sizes'] = seg_data.shape

    # create the array
    affine = np.vstack((np.array([np.nan, np.nan, np.nan]), im_hdr['space directions']))
    hdr['space directions'] = affine
    hdr['kinds'] = ['list', 'domain', 'domain', 'domain']
    hdr['encoding'] = 'gzip'
    hdr['space origin'] = im_hdr['space origin']
    hdr['measurement frame'] = seg_hdr['measurement frame']

    #color for the seg
    values = np.unique(seg_data[seg_data != 0])
    #norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    norm = mpl.colors.Normalize(vmin=bnds[0], vmax=bnds[1])
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet_r)
    cmap.set_array([])

    # for element set the cost value which
    for idx in range(seg_data.shape[0]):
        root = 'Segment{}'.format(idx)

        #get color for score
        shape = seg_data[idx, :, :, :]

        score = 1

        color = cmap.to_rgba(score)
        #hdr['{}_Color'.format(root)] = 'R={0:.3f} G={1:.3f} B={2:.3f} A=1.000'.format(color[0], color[1], color[2])
        hdr['{}_Color'.format(root)] = '{0} {1} {2}'.format(color[0], color[1], color[2])
        hdr['{}_ColorAutoGenerated'.format(root)] = '0'
        X, Y, Z = np.where(shape != 0) # get min and max
        hdr['{}_Extent'.format(root)] = '{0} {1} {2} {3} {4} {5}'.format(X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max())
        hdr['{}_ID'.format(root)] = 'Segment_{}'.format(idx)
        hdr['{}_Name'.format(root)] = 'Cost: {}'.format(score)
        hdr['{}_NameAutoGenerated'.format(root)] = '0'
        hdr['{}_Tags'.format(root)] = 'TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SRT^T-D0050^Tissue~SRT^T-D0050^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'


    hdr['Segmentation_ContainedRepresentationNames'] = seg_hdr['Segmentation_ContainedRepresentationNames']
    hdr['Segmentation_ConversionParameters'] = seg_hdr['Segmentation_ConversionParameters']
    hdr["Segmentation_MasterRepresentation"] = seg_hdr["Segmentation_MasterRepresentation"]
    hdr['Segmentation_ReferenceImageExtentOffset'] = seg_hdr['Segmentation_ReferenceImageExtentOffset']

    return hdr

def create_crani_heatmap(fname, geometry):
    """
    This will be the heatmap -
    score each craniotomy position on a scale based on the number of all the possible points of the target it can access

    :param fname:
    :param geometry:
    :return:
    """

    df = pd.read_csv(fname)
    df = correct_df(df)

    # get the list of the craniotomy coordinates and all their respective scores for each target
    crani = df.loc[0, 'crani']
    scores = np.stack(df['cost'].values)

    # iterate thru each crani coordinate and get the score for each of the targets -- then put in outarray
    score_list = np.empty((0,4))
    seg = []
    for idx in range(crani.shape[0]):

        im = np.zeros(([geometry[0], geometry[1], geometry[2] +3]))

        # find all the scores for the crani position and all respective target location
        crani_vals = scores[:, idx]
        inf = crani_vals != np.inf # find whihc of the vals are not infinite
        frac = np.sum(inf)/len(crani_vals) + 0.01 # frac score =  infinte/total | 1 is 100% valid pth: 0 is 0% valid pth

        # add data to out array
        row = np.concatenate((crani[idx, :], [frac]))
        score_list = np.vstack((score_list, row))

        im[crani[idx, 0], crani[idx, 1], crani[idx, 2]] = frac
        im = binary_dilation(im, iterations=3).astype(int) * frac
        seg.append(im)


    # create 4D image
    segmentation = np.stack(seg, axis=0)

    return segmentation

def create_target_heatmap(fname, geometry):
    """ creates heatmap for the target based on how many paths can travel to the point"""

    df = pd.read_csv(fname)
    df = correct_df(df)

    seg = []
    for idx in df.index.values:

        im = np.zeros(geometry)
        # can display number of paths that are valid
        target = df.loc[idx, 'target']
        scores = df.loc[idx, 'cost']
        heat = np.sum(scores != np.inf)/ len(scores) + 0.01
        im[target[0], target[1], target[2]] = 1
        im[binary_dilation(im)] = heat

        seg.append(im)

    # create 4d image
    segmentation = np.stack(seg, axis=0)

    return segmentation

def create_spider_map(fname, target, geometry, spacing, crani_pos = []):
    """
    Creates the spider map with each of the entrypoints of the craniotomy and the target
    :param
    :return:
    """


    seg = [] # give option to run the code when crani_pos is defined
    if len(crani_pos) >= 1:

        cone_im = np.zeros(geometry)
        spacing = np.array([1, 1, 1])
        cone = Cylinder(crani_pos, target, limit_dict['crani_radius'], limit_dict['crani_radius'] / 2,
                        limit_dict['cyl_radius'], geometry, spacing)
        cone.create_shape()
        cone_im[np.where(cone.voxel == 1)] = 1
        seg.append(cone_im)

        # create 4D image
        segmentation = np.stack(seg, axis=0)

        return segmentation

    df = pd.read_csv(fname)
    df = correct_df(df)

    for row in df.iterrows():
        if np.array_equal(row[1]['target'], target):
            row = row[1]
            break

    # iterate through each crani point and create a cylinder
    for idx, crani in enumerate(row['crani']):

        cone_im = np.zeros(geometry)
        cone = Cylinder(crani, target, limit_dict['crani_radius'], limit_dict['crani_radius']/2,
                        limit_dict['cyl_radius'], geometry, spacing)
        cone.create_shape2()
        score = row['cost'][idx]
        if score == np.inf: score = 99999
        cone_im[np.where(cone.voxel == 1)] = row['cost'][idx]
        seg.append(cone_im)

    # create 4D image
    segmentation = np.stack(seg, axis=0)

    return segmentation


# set tunable variables
wd = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\'
pt = '15'

for app in ['MCF']:
    # set the variables for funsie
    im_pth = join(wd, 'pt_{}'.format(pt), 'pt_{}_padded.nrrd'.format(pt))
    seg_pth = glob(join(wd, 'pt_{0}\\?t_{0}_?egmentation.seg.nrrd'.format(pt)))[0]
    csv = join(wd, 'pt_{0}\\pt_{0}_{1}_ero3.csv'.format(pt, app))

    # set the variables for outputting files
    chm_out = join(wd, 'pt_{0}\\chm_{1}.nrrd'.format(pt, app))
    thm_out = join(wd, 'pt_{0}\\thm_{1}.nrrd'.format(pt, app))

    seg, seg_hdr = nrrd.read(seg_pth)
    im, hdr = nrrd.read(im_pth)
    img_spacing = np.linalg.norm(hdr['space directions'],
                                 axis=1)  # slice spacing is defined as the normalized vector of the space transformation matrix
    geo = im.shape

#    # create_target_heatmap(csv, geo)
#    thm = create_target_heatmap(csv, geo)
#
#    #TODO delete this jank shit
#    #This is a work around to put all the points in one array instead of have [21, X, Y, Z] array. This saves memory
#    thm = np.array([np.sum(thm, axis=0)])
#
#    thm_hdr = create_seg_hdr(hdr, seg_hdr, thm, [0, 1.0])
#    thm[np.where(thm != 0)] = 1
#    nrrd.write(thm_out, thm, thm_hdr)

    spider = create_spider_map(csv, np.array([218, 327, 42]), geo, img_spacing, crani_pos=np.array([147, 438, 23]))
    myhdr = create_seg_hdr(hdr, seg_hdr, spider, [0,200])
    nrrd.write(join(wd, 'pt_{0}\\spider_test.nrrd'.format(pt, app)), spider.astype('uint8'), myhdr)

    spider = create_spider_map(csv, np.array([72, 151, 58]), geo, img_spacing)
    myhdr = create_seg_hdr(hdr, seg_hdr, spider, [200, 2000])
    nrrd.write(join(wd, 'pt_{0}\\spider_{1}.nrrd'.format(pt, app)), spider.astype('uint8'), myhdr)


#    spider = create_spider_map(csv, np.array([275, 312, 200]), geo, img_spacing)
#    for i in range(spider.shape[0]):
#        spider_1d = np.array([spider[i, :, :, :]])
#        myhdr = create_seg_hdr(hdr, seg_hdr, spider_1d, [200, 2000])


    chm = create_crani_heatmap(csv, geo)
    chm_hdr = create_seg_hdr(hdr, seg_hdr, chm, [0, 1.0])
    chm[np.where(chm != 0)] = 1
    nrrd.write(chm_out, chm, chm_hdr)
