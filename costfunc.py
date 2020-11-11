import numpy as np
import json
import nrrd
import pandas as pd
from utils import array_to_coords, binarize_segmentation, NumpyEncoder
from coordinates import find_edge_points, create_crani_grid
from weights import LSB_class_weights, LSB_class_group, microscope_dict, dist_map_dict, limit_dict
from surgicalpath import Cylinder
from scipy.ndimage import generate_binary_structure
from collections import OrderedDict

class CostFunc:
    """
    Generalized class that can be used to create a cost function and evaluate the path
    the cost function then can be used to save potential paths that are low in score

    """

    def __init__(self, entry_point, seg_data, lut, dist_maps, def_dict, img_spacing,
                 weight_table=LSB_class_weights, group_table=LSB_class_group, dist_dict=dist_map_dict,
                 lim_dict=limit_dict):


        # create the target coords
        target_struct = binarize_segmentation(seg_data, lut["Target"])
        target_coords = array_to_coords(target_struct)

        # assign values
        self.lut = lut
        self.entry_point = entry_point
        self.target_coords = target_coords
        self.dist_maps = dist_maps
        self.img_spacing = img_spacing
        self.weight_table = weight_table
        self.group_table = group_table
        self.dist_dict = dist_dict
        self.seg_data = seg_data
        self.lim_dict = lim_dict

        # create the removable structures_dict
        remov_name = [k for k in lut.keys() if group_table[k]['Class'] == 'Removable']
        remov_dict = {}
        for name in remov_name:
            remov_struct = binarize_segmentation(self.seg_data, lut[name])
            remov_dict[name] = remov_struct
        self.remov_dict = remov_dict

        # create critical structures array
        crit = [k for k in lut.keys() if group_table[k]['Class'] == 'Critical']
        crit_struct = binarize_segmentation(self.seg_data, [lut[name] for name in crit])
        self.crit_struct = crit_struct

        # create the deformable structures dictionary
        corr_def_dict = {}
        for group, map in def_dict.items():
             def_grad = map

             # create a binary gradient map to use as a scalar
             grad_scalar = np.zeros(def_grad.shape)
             grad_scalar[np.where(def_grad != 0)] = self.group_table[group]["Weight"]

             # create the gradient step
             def_grad = def_grad - 1
             def_grad[np.where(def_grad == -1)] = 0
             def_grad = def_grad * 0.1 #this is the step size

             def_grad = grad_scalar + def_grad
             corr_def_dict[group] = def_grad

        self.def_dict = corr_def_dict

    def cyl_cost(self, cyl):
        """ get the cost of the cylinder traveling through the deformables and the removable structure

        :param cyl -- voxelization of the cylinder
        :returns costvalue (scalar for cost); out_dict dictionary with [group and score and n# voxels]
        """

        from time import time

        # data are stored in the format  ["class": [score, num of voxels]]
        out_dict = {}

        # get deformable scores
        def_group = [k for k, v in self.group_table.items() if v["Class"] == "Deformable"]
        for group in def_group:
            def_grad = self.def_dict[group]
            def_intersect = check_intersect(def_grad, cyl)
            out_dict[group] = [def_intersect, np.sum(np.logical_and(def_grad, cyl))]

        # get removable structures
        for name, remov_struct in self.remov_dict.items():
            remov_intersect = check_intersect(remov_struct, cyl)
            out_dict[name] = [remov_intersect * self.group_table[name]["Weight"], remov_intersect]

        # get critical structure weights
        crit_intersect = check_intersect(self.crit_struct, cyl)
        out_dict['Critical'] = [crit_intersect*1000, crit_intersect] # weights for crit structure is here

        costvalue = np.sum([v[0] for v in out_dict.values()])

        return costvalue, out_dict

    def limit_cost(self):

        # find the target coordinates
        target = binarize_segmentation(self.seg_data, self.lut["Target"])
        struct = generate_binary_structure(3, 1)
        target_pts = find_edge_points(target, struct)

        # find the craniotomy coordinates
        crani_pts = create_crani_grid(self.entry_point, stepnum=3, offst=2)

        # create df which will store all data
        target_df = pd.DataFrame()

        # map each target to each craniotomy
        for tar_idx, tar_pt in enumerate(target_pts[:1]):

            tar_pt = np.array([260, 282, 33])

            crani_df = pd.DataFrame()

            for crani_idx, crani_pt in enumerate(crani_pts):

                crani_pt = np.array([376, 209, 73])

                # create 1 voxel cylinder which represents minimum path
                cyl = Cylinder(crani_pt, tar_pt,
                               self.lim_dict['cyl_radius'], self.lim_dict['cyl_radius'], self.lim_dict['cyl_radius'],
                               self.seg_data.shape, self.img_spacing)
                cyl.create_shape(num=5)

                # create the cone which represents costs
                cone = Cylinder(crani_pt, tar_pt,
                                self.lim_dict['crani_radius'], self.lim_dict['crani_radius']/2, self.lim_dict['cyl_radius'],
                                self.seg_data.shape, self.img_spacing)
                cone.create_shape(num=20)

                # create a dataframe to store the data
                ser_df = pd.DataFrame()
                ser_df['targ'] = [tar_pt]
                ser_df['crani'] = [crani_pt]


                # calculate the cost of the cylinder through the deformable
                cost, cost_dict = self.cyl_cost(cone.voxel)

                if np.any(np.logical_and(self.crit_struct, cyl.voxel)):
                    cost = np.inf

                ser_df['cost'] = cost
                for key, value in cost_dict.items():
                    ser_df[key] = [value]

                """
                #find the miminimum dist by finding the minimum intersection between distance map and the cylinder
                intersect = self.dist_maps['Dist_Coch_SCC'][np.where(cyl.voxel == 1)]
                min_dist = np.min(intersect)
                ser['min_dist'] = [min_dist]
                
                # calculate the score with the formula
                # score = cost - cost * w_dist * log(dist) or simplified to score = cost(1 - w_dist * log(dist))
                # added np.log2(min_dist + 1) for two reasons
                # 1) we want the highest cost to be for when the min_dist == 0, which will be then just defined as score = np.inf
                # 2) base 2 log has a greater slope, which gives a greater weighting for increases in min_dist as we go to larger values
                w_dist = self.lim_dict['w_dist']

                # see if the cylinder intersects with any of our critical structures
                if np.any(np.logical_and(self.crit_struct, cyl.voxel)):
                    min_dist = 0
                if min_dist == 0:
                    ser['score'] = np.inf
                else:
                    ser['score'] = cost*(1 - w_dist * np.log2(min_dist + 1))
                """

                # add to the crani_df
                crani_df = pd.concat((crani_df, ser_df))

            crani_df = crani_df.reset_index().drop(['index'], axis=1)

            tar_df = pd.DataFrame()
            tar_df['target'] = [tar_pt]
            tar_df['min_cost'] = [crani_df['cost'].min()]
            tar_df['crani'] = [crani_df['crani'].values]
            tar_df['cost'] = [crani_df['cost'].values]
            for key in cost_dict.keys():
                tar_df[key] = [crani_df[key].values]


            target_df = pd.concat((target_df, tar_df))

        #TODO write the JSON dump

        # score is the total cost
        target_df = target_df.reset_index().drop(['index'], axis=1)
        self.tar_df = target_df

    def save_data(self, pth):

        #get root
        root = pth.split('.csv')[0]

        # save the weights
        weights = {}
        weights['weight table'] = self.weight_table
        weights['group table'] = self.group_table
        weights['limit dict'] = self.lim_dict
        with open(root+'_weights.json', 'w') as fp:
            json.dump(weights, fp, cls=NumpyEncoder, indent=4, sort_keys=True)

        with open(pth, 'w') as fp:
            json.dump(self.tar_dict, fp, cls=NumpyEncoder, indent=4, sort_keys=True)


    def save_dataframe(self, path):
        """function to save the dataframe as a csv so that it can be accessed again for computation"""

        #get root
        root = path.split('.csv')[0]

        # save the weights
        weights = {}
        weights['weight table'] = self.weight_table
        weights['group table'] = self.group_table
        weights['limit dict'] = self.lim_dict
        with open(root+'_weights.json', 'w') as fp:
            json.dump(weights, fp, cls=NumpyEncoder, indent=4, sort_keys=True)

        #save the weight dictionaries for reference
        df = self.tar_df
        df.to_csv(path, index=False)

    def evaluate_cost(self):
        """
        Evaluates the cost function based on tissue classes
        :return: costvalue
        """

        # return an inferior costvale for the function if it intersects any critical structures
        crit_stuct = binarize_segmentation(self.seg_data, [self.lut[s] for s,v in self.lut.items() if self.group_table[s]["Class"] == "Critical"])
        if check_intersect(crit_stuct, self.cyl):
            print("intersection with crit structures")
            costvalue = np.inf
            self.cost = costvalue
            return costvalue

        # get distance maps
        dist_scores = []
        for name, dist_map in self.dist_maps.items():
            intersect = dist_map*self.cyl
            min_dist = np.min(intersect[intersect > 0])
            if min_dist < self.dist_dict[name]["Min_Distance"]:
                costvalue = np.inf
                self.cost = costvalue
                return costvalue
            dist_scores.append(min_dist * self.weight_table[name]["Factor"]) #TODO give scalar value to increase the weight of distance scores

        self.micro_vtx, self.micro = SurgicalPath.create_microscope(self.entry_point, self.target_coords,
                                                                    self.seg_data.shape, self.img_spacing)

        # get deformable scores
        def_scores = []
        def_group = [k for k,v in self.group_table.items() if v["Class"] == "Deformable"]
        for group in def_group:
            def_grad = self.def_dict[group]
            def_grad = self.weight_table[group]["Factor"] + def_grad * (1/10)
            def_intersect = check_intersect(def_grad, self.micro)
            def_scores.append(def_intersect)

        # get removable structures
        remov_group = ["Tissue", "Bone"]
        remov_scores = []
        for group in remov_group:
            remov_struct = binarize_segmentation(self.seg_data,
                                                 [self.lut[s] for s in self.lut.keys() if self.group_table[s]["Group"] == group])
            remov_intersect = check_intersect(remov_struct, self.micro)
            remov_scores.append(remov_intersect *
                                          self.weight_table[group]["Factor"])

        # get target
        target_struct = binarize_segmentation(self.seg_data, self.lut["Target"])
        target_intersect = check_intersect(target_struct, self.micro)
        target_score = target_intersect * self.weight_table["Target"]["Factor"]

        costvalue = sum(def_scores + remov_scores + dist_scores + target_score)
        self.cost = costvalue
        return costvalue

    def save_paths(self, hdr, fextension):
        """
        save the  paths as images

        :param hdr: nrrd hdr for the outputted saved images
        :param fextension: file extension; example 'C:\\data\\LSB\\pt_6\\pt_6_'
        :return:
        """

        if fextension[-1] != '_':
            fextension = fextension + '_'

        cyl_fname = fextension + 'cyl.nrrd'
        micro_fname = fextension + 'micro.nrrd'

        nrrd.write(cyl_fname, self.cyl, hdr)

        if self.micro:
            nrrd.write(micro_fname, self.micro, hdr)

def check_intersect(array1, array2):
    """ return sum of all intersecting points of two arrays
        multiplied because it allows for multiplacation ofo binary mask with gradient and binary mask with binary mask"""
    return np.sum(array1 * array2)