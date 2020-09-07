from numpy import inf
# all values are in mm
microscope_dict = {"Radius": 35,
                   "Focal Length": 300,
                   "Min Distal Radius": 1.5}

limit_dict = {"w_dist": 0.0, # weight for how much min_distance effects the cost function in the costfunc limit_cost() method
              "cyl_radius": 0.25, # the radius for the cylinders evaluating the limit_cost() function
              "crani_radius": 5}

""" LSB class weights:
These are the weights for the different structures -- trying to give amore comprehensive way to score their values
"""

LSB_class_weights = {"Brain": {"Factor": 1.0, "Limits": [0.0, 0.5]},
                     "Sigmoid_Sinus": {"Factor": 1.0, "Limits": [0.0, 0.5]},
                     "Bone": {"Factor": 1.0, "Limits": [0.1, 0.15]},
                     "Tissue": {"Factor": 1.0, "Limits": [0.1, 0.15]},
                     "Target": {"Factor": -1.0, "Limits": [0.01, 0.2]},
                     "Dist_Coch_SCC": {"Factor": -1.0, "Limits": [0.0, 50.0]},
                     "Dist_Vasc": {"Factor": -1.0, "Limits": [0.0, 50.0]}, #TODO these are not calibrated to image dimensions
                     "Dist_Brain_Stem": {"Factor": -1.0, "Limits": [0.0, 50.0]}}


"""
 LSB_class_group
 allows for classification of each of the segmentation tissue types into different overarching categories.
 each segmentaiton is part of a class and group class
 
 Classes are the 4 main groups that a segmentation could go into
 Deformable: a structure that is deformable and will be scored with a deformability gradient
 Removable: a structure that can be removed during surgery w/ minimal harm (tissue or bone) 
 Critical: a structure that cannot be removed during surgery and minimum distance must be maintained
 Entry: a potential entry point for a craniotomy (depreciated)
 
"""
LSB_class_group = {
                "Background":         {"Class": "Noise",
                                       "Group": "Noise"},
                "Brain":              {"Class": "Deformable",
                                       "Group": "Brain",
                                       "Weight": 0.15},
                "Cerebellum":         {"Class": "Deformable",
                                       "Group": "Cerebellum",
                                       "Weight": 0.1},
                "Bone":               {"Class": "Removable",
                                       "Group": "Bone",
                                       "Weight": 0.2},
                "Mastoid":            {"Class": "Removable",
                                       "Group": "Bone",
                                       "Weight": 0.15},
                "Tissue":             {"Class": "Removable",
                                       "Group": "Tissue",
                                       "Weight": 0.15},
                "Superficial_Tissue": {"Class": "Removable",
                                       "Group": "Tissue",
                                       "Weight": 0.15},
                "EAC":                {"Class": "Removable",
                                       "Group": "Tissue",
                                       "Weight": 0.05},
                "Target":             {"Class": "Target",
                                       "Group": "Target",
                                       "Weight": 3.0},
                "PSCC":               {"Class": "Critical",
                                       "Group": "Coch_SCC",
                                       "Weight": 500},
                "LSCC":               {"Class": "Critical",
                                       "Group": "Coch_SCC",
                                       "Weight": 500},
                "SSCC":               {"Class": "Critical",
                                       "Group": "Coch_SCC",
                                       "Weight": 500},
                "Vestibule":          {"Class": "Critical",
                                       "Group": "Coch_SCC",
                                       "Weight": 1000},
                "Cochlea":            {"Class": "Critical",
                                       "Group": "Coch_SCC",
                                       "Weight": 1000},
                "Middle_Ear":         {"Class": "Removable",
                                       "Group": "Removable",
                                       "Weight": 0},
                "Sigmoid_Sinus":      {"Class": "Removable",
                                       "Group": "Sigmoid_Sinus",
                                       "Weight": 0.45},
                "Carotid":            {"Class": "Critical",
                                       "Group": "Vasc",
                                       "Weight": 1000},
                "Jugular":            {"Class": "Critical",
                                       "Group": "Vasc",
                                       "Weight": 1000},
                "Brain_Stem":         {"Class": "Critical",
                                       "Group": "Brain_Stem",
                                       "Weight": 1000},
                "MCF":                {"Class": "Entry",
                                       "Group": "MCF"},
                "MCF_middle":         {"Class": "Entry",
                                       "Group": "MCF"},
                "RS":                 {"Class": "Entry",
                                       "Group": "RS"},
                "RS_middle":          {"Class": "Entry",
                                       "Group": "RS"}
                           }


"""
These are weights for the minimum distance maps that can be used
 it finds all the segmentations that is within the group and then sets the maximum distance and the minimum distance
 in mm -- this is then converted to voxel dimensions by the image spacing """

dist_map_dict = {
#                 "Dist_Coch_SCC": {"Class": "Distance",
#                              "Group": "Coch_SCC",
#                              "Max_Distance": 5, #distance in mm
#                              "Min_Distance": 3}, #distance in mmm
#                 "Dist_Vasc": {"Class": "Distance",
#                               "Group": "Vasc",
#                               "Max_Distance": 5, # distance in mm
#                               "Min_Distance": 3}, # distance in mm
#                 "Dist_Brain_Stem": {"Class": "Distance",
#                                "Group": "Brain_Stem",
#                                "Max_Distance": 1, # distance in mm
#                                "Min_Distance": 3} #distance in mm
                 }

def get_group_members(group, mydict):
    """ get members of a group from a weight dictionary"""
    group_list = []
    for k, v in mydict.items():
        if v["Group"] == group:
            group_list.append(k)
    return group_list

def get_class_members(class_name, mydict):
    """ get members of a class from a weight dictionary"""
    class_list = []
    for k, v in mydict.items():
        if v["Class"] == class_name:
            class_list.append(k)
    return class_list