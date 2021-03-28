from utils import find_center, array_to_coords, binarize_segmentation
from weights import LSB_class_group, LSB_class_weights, dist_map_dict, microscope_dict
from surgicalpath import SurgicalPath
import numpy as np
from time import time
from scipy.ndimage import binary_erosion


class CostFunction:

    @staticmethod
    def costfunct_dictionary(crani_coords, test_points, seg_data, lut, dist_maps, def_dict, img_spacing,
                             weight_table=LSB_class_weights, group_table=LSB_class_group):
        """

        :param crani_coords: coordinates of the center of the craniotomy shape
        :param seg_data:  segmentation data
        :param lut:  look up table that maps the strings of the tissues to the integer value they are coded for in seg_data
        :param dist_maps: distance mapfrom the structures
        :param img_spacing: dimensions of the image voxels in real mm
        :param weight_table: table with the weights of each class and structure
        :param group_table: group table that classicfies and organizes which weights and structures belong where
        :return:
        """
        print('guess coordinates {}'.format(crani_coords))
        crani_coords = np.array([np.round(x) for x in crani_coords])
        if any(np.isnan(crani_coords)):
            costvalue = np.inf
            return costvalue
        if not any([all(crani_coords == p) for p in test_points]):
            costvalue = np.inf
            return costvalue

        target_struct = binarize_segmentation(seg_data, lut["Target"])
        target_coords = array_to_coords(target_struct)


        # create the different surgical paths
        s = time()
        cyl_vertex, cyl_path = SurgicalPath.create_bicone_cylinder(crani_coords, target_coords,
                                                                   microscope_dict["Min Distal Radius"], microscope_dict["Min Distal Radius"], 1,
                                                                   seg_data.shape, img_spacing)
        s2 = time()
        print("took {} s to create bicone cylinder".format(s2-s))

        # return an inferior costvale for the function if it intersects any critical structures
        t1 = time()
        crit_stuct = binarize_segmentation(seg_data, [lut[s] for s,v in lut.items() if group_table[s]["Class"] == "Critical"])
        if check_intersect(crit_stuct, cyl_path):
            print("intersection with crit structures")
            costvalue = np.inf
            return costvalue
        print("took {}s to check crit structures FAST".format(time() - t1))

        # get distance maps
        dist_scores = []
        for name, dist_map in dist_maps.items():
            intersect = dist_map*cyl_path
            min_dist = np.min(intersect[intersect > 0])
            if min_dist < dist_map_dict[name]["Min_Distance"]:
                costvalue = np.inf
                return costvalue
            dist_scores.append(min_dist *
                                   weight_table[name]["Factor"]) #TODO give scalar value to increase the weight of distance scores


        s2 = time()
        micro_vtx, micro_path = SurgicalPath.create_microscope(crani_coords, target_coords, seg_data.shape, img_spacing)
        surgical_path = [micro_path, cyl_path]
        s = time()
        print("took {}s to create surgical path".format(s-s2))


        # get deformable scores
        def_scores = []
        def_group = [k for k,v in group_table.items() if v["Class"] == "Deformable"]
        for group in def_group:
            #def_struct = binarize_segmentation(seg_data, [lut[s] for s in lut.keys() if group_table[s]["Group"] == group])
            #def_grad = create_erosion_gradient(def_struct, img_spacing)
            def_grad = def_dict[group]
            def_grad = weight_table[group]["Factor"] + def_grad * (1/10)
            def_intersect = check_intersect(def_grad, micro_path)
            def_scores.append(def_intersect)
        s2 = time()
        print("took {}s to create deformation".format(s2- s))

        # get removable structures
        remov_group = ["Tissue", "Bone"]
        remov_scores = []
        for group in remov_group:
            remov_struct = binarize_segmentation(seg_data, [lut[s] for s in lut.keys() if group_table[s]["Group"] == group])
            remov_intersect = check_intersect(remov_struct, micro_path)
            remov_scores.append(remov_intersect *
                                          weight_table[group]["Factor"])
        s = time()
        print("took {} s to create removable intersections".format(s-s2))


        s2 = time()
        # get target
        target_intersect = check_intersect(target_struct, micro_path)
        target_score = target_intersect * weight_table["Target"]["Factor"]
        s = time()
        print("took {} s to get target scores".format(s - s2))

        costvalue = sum(def_scores + remov_scores + dist_scores + target_score)
        return costvalue

    @staticmethod
    def costfunc_disc_crani_normalized(crani_coords,
                                       brain, bstem, bone, target, coch_scc, vasc, tissue,
                                       coch_scc_dist, vasc_dist, bstem_dist, img_spacing):

        """
        cost function for the disc with crani normalized

        :param crani_coords: [[x1, y1, z1], [x2, y2, z2],...]
        :param brain: voxel array
        :param bstem: voxel array
        :param bone: voxel array
        :param target: voxel array
        :param coch_scc:  voxel array
        :param vasc: voxel array
        :param tissue: voxel array
        :param coch_scc_dist:  distance map
        :param vasc_dist:  distance map
        :param bstem_dist: distance map
        :param img_spacing: voxel dimensions of the image
        :return:  costvalue (float)
        """

        target_coords = array_to_coords(target)

        # create the crani intersects and the cylinder
        # crani will be the general shape and the cylinder will represent the actual surgical path

        crani_vertex, crani_path = SurgicalPath.create_disc_with_crani(crani_coords, target_coords, target.shape,
                                                                             img_spacing)
        cyl_vertex, cyl_path = SurgicalPath.create_bicone_cylinder(crani_coords, target_coords, 1, 1, 1,
                                                                   target.shape, img_spacing)
        surgical_path = [crani_path, cyl_path]

        if check_intersect(coch_scc, cyl_path):
            _error = 'intersection with coch_scc at {}'.format(np.where(np.logical_and(coch_scc, crani_path)))
            print(_error)
            costvalue = np.inf
            return costvalue

        if check_intersect(bstem, cyl_path):
            _error = 'intersection with bstem at {}'.format(np.where(np.logical_and(bstem, crani_path)))
            print(_error)
            costvalue = np.inf
            return costvalue

        if check_intersect(vasc, cyl_path):
            _error = 'intersection with vasc at {}'.format(np.where(np.logical_and(vasc, crani_path)))
            print(_error)
            costvalue = np.inf
            return costvalue

        # create deformability map for cylinder as defined by the anatomical plane.



        # find intersection of structures and the bicone path
        path_bone = check_intersect(cyl_path, bone)
        path_tissue = check_intersect(cyl_path, tissue)
        path_target = check_intersect(cyl_path, target)

        coch_scc_min_dist = get_min_distance(coch_scc_dist, crani_path)
        vasc_min_dist = get_min_distance(vasc, vasc_dist)
        bstem_min_dist = get_min_distance(bstem, bstem_dist)

        # log normalization from Nava's code

        volume_crani = np.sum(crani_path)
        volume_cyl = np.sum(cyl_path)

        weights = {"Brain": [0.0, 0.5], "Bone": [0.1, 0.8], "Tissue": [0.0, 0.5], "Target": [0, 0.1],
                   "Distance Coch SCC": [0, 50], "Distance Vasc": [0, 50], "Distance Bstem": [0, 50]}

        min_brain = 0
        max_brain = 4000  # deformable
        min_bone = 1000
        max_bone = 10000  # bone
        min_tissue = 0
        max_tissue = 4000  # tissue
        min_target = 0
        max_target = 1000  # target
        min_dis_coch_scc = 0  # cochlea - semicircular anals
        max_dis_coch_scc = 50
        min_dis_vasc = 0  # vasculatyre
        max_dis_vasc = 50
        min_dis_bstem = 0  # bstem
        max_dis_bstem = 50

        brain_nrm = normalize(path_bone, min_brain, max_brain)
        bone_nrm = normalize(path_bone, min_bone, max_bone)
        tissue_nrm = normalize(path_tissue, min_tissue, max_tissue)
        target_nrm = normalize(path_target, min_target, max_target)
        min_dis_coch_scc_nrm = normalize(coch_scc_min_dist, min_dis_coch_scc, max_dis_coch_scc)
        min_dis_vasc_nrm = normalize(vasc_min_dist, min_dis_vasc, max_dis_vasc)
        min_dis_bstem = normalize(bstem_min_dist, min_dis_bstem, max_dis_bstem)

        costvalue = 1 * brain_nrm + bone_nrm + tissue_nrm - target_nrm - min_dis_coch_scc_nrm \
                        - min_dis_vasc_nrm - min_dis_bstem

        return costvalue


def check_intersect(array1, array2):
    """ return sum of all intersecting points of two arrays
        multiplied because it allows for multiplacation ofo binary mask with gradient and binary mask with binary mask"""
    return np.sum(array1 * array2)


def get_min_distance(distancemap, path):
    """ return minimum value of intersection between distancemap and path"""
    return np.min(distancemap[np.where(path == 1)])

def normalize(value, limits):
    """ returns the log2 normalized value"""
    low, high = limits
    if (value-low)/(high-low) <= 0:
        return 0
    else:
        return np.log2((value-low) / (high-low))


