import numpy as np
from weights import microscope_dict

def pad_seg_data(seg_data, img_spacing, micro_dict=microscope_dict, ax=2):
    """
    Pads the segmentation data in the specific axis dimension

    :param seg_data: segmentation data
    :param img_spacing:
    :param mico_dict: dictionary for the microscope data
    :param ax: the axis along which we pad the image
    :return:
    """

    pad = int(micro_dict["Radius"]/img_spacing[ax]/2)

#    pad_im = np.hstack((seg_data, np.tile(seg_data[:, :, [-1]], pad)))
    pad_im = np.repeat(seg_data, [1]*(seg_data.shape[ax]-1) + [pad], axis=ax)

    return pad_im

