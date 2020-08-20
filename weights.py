# all values are in mm
microscope_dict = {"Radius": 35,
                   "Focal Length": 300,
                   "Min Distal Radius": 1.5}

LSB_class_weights = {"Brain": {"Factor": 1.0, "Limits": [0.0, 0.5]},
                     "Sigmoid_Sinus": {"Factor": 1.0, "Limits": [0.0, 0.5]},
                     "Bone": {"Factor": 1.0, "Limits": [0.1, 0.15]},
                     "Tissue": {"Factor": 1.0, "Limits": [0.1, 0.15]},
                     "Target": {"Factor": -1.0, "Limits": [0.01, 0.2]},
                     "Dist_Coch_SCC": {"Factor": -1.0, "Limits": [0.0, 50.0]},
                     "Dist_Vasc": {"Factor": -1.0, "Limits": [0.0, 50.0]}, #TODO these are not calibrated to image dimensions
                     "Dist_Brain_Stem": {"Factor": -1.0, "Limits": [0.0, 50.0]}}

LSB_class_group = {
                "Background":         {"Class": "Noise",
                                       "Group": "Noise"},
                "Brain":              {"Class": "Deformable",
                                       "Group": "Brain"},
                "Bone":               {"Class": "Removable",
                                       "Group": "Bone"},
                "Mastoid":            {"Class": "Removable",
                                       "Group": "Bone"},
                "Tissue":             {"Class": "Removable",
                                       "Group": "Tissue"},
                "Superficial_Tissue": {"Class": "Removable",
                                       "Group": "Tissue"},
                "EAC":                {"Class": "Removable",
                                       "Group": "Tissue"},
                "Target":             {"Class": "Removable",
                                       "Group": "Target"},
                "PSCC":               {"Class": "Critical",
                                       "Group": "Coch_SCC"},
                "LSCC":               {"Class": "Critical",
                                       "Group": "Coch_SCC"},
                "SSCC":               {"Class": "Critical",
                                       "Group": "Coch_SCC"},
                "Vestibule":          {"Class": "Critical",
                                       "Group": "Critical"},
                "Cochlea":            {"Class": "Critical",
                                       "Group": "Coch_SCC"},
                "Middle_Ear":         {"Class": "Critical",
                                       "Group": "Coch_SCC"},
                "Sigmoid_Sinus":      {"Class": "Removable",
                                       "Group": "Sigmoid_Sinus"},
                "Carotid":            {"Class": "Critical",
                                       "Group": "Vasc"},
                "Jugular":            {"Class": "Critical",
                                       "Group": "Vasc"},
                "Brain_Stem":         {"Class": "Critical",
                                       "Group": "Brain_Stem"},
                "MCF":                {"Class": "Entry",
                                       "Group": "MCF"},
                "MCF_middle":         {"Class": "Entry",
                                       "Group": "MCF"},
                "RS":                 {"Class": "Entry",
                                       "Group": "RS"},
                "RS_middle":          {"Class": "Entry",
                                       "Group": "RS"}
                           }



dist_map_dict = {"Dist_Coch_SCC": {"Class": "Distance",
                              "Group": "Coch_SCC",
                              "Max_Distance": 5, #distance in mm
                              "Min_Distance": 3}, #distance in mmm
                 "Dist_Vasc": {"Class": "Distance",
                               "Group": "Vasc",
                               "Max_Distance": 5, # distance in mm
                               "Min_Distance": 3}, # distance in mm
 #                "Dist_Brain_Stem": {"Class": "Distance",
 #                               "Group": "Brain_Stem",
 #                               "Max_Distance": 1, # distance in mm
 #                               "Min_Distance": 3} #distance in mm
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