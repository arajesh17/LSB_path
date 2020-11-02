import nrrd
from utils import binarize_segmentation
import numpy as np
from scipy.ndimage import binary_dilation

class LoadData:

    @staticmethod
    def load_segmentation(pth, lookuptable):
        """
        loads segmentation and binarizes it to return

        :param self:
        :param pth: is the path to the segmentation in nrrd format
        :param lookuptable:  is the path to the lookup table for the labeling of nrrd segmentation
        :return:  [brain_seg, bone_seg, air_seg, tumor_seg, nerve_seg, cochlea_seg]
        """
        data, hdr = nrrd.read(pth)

        with open(lookuptable) as f:
            label_dict = {}
            for line in f.readlines()[2:]:
                line = line.split(' ')
                label_dict[line[1]] = int(line[0])

        return data, label_dict

    @staticmethod
    def load_image(pth):
        data, hdr = nrrd.read(pth)
        return data, hdr

    @staticmethod
    def save_image(pth: object, img: object, hdr: object) -> object:
        nrrd.write(pth, img, hdr)

    @staticmethod
    def create_distance_map(segmentation, n_iter, kernel = np.ones((3,3,3))):
        """
        trick to create a distance map quickly -- dilat the shape by number of interations
        until you create an approximate distance map - every distance more than the n_iter is by
        default stored as the n_iter value

        :param segmentation: segmentation of the image
        :param n_iter: number of iterations to dilate -- will be max distance recorded
        :return: kernel -- shape of the kernel to dilate by
        """

        dilation_mask = np.zeros(segmentation.shape)

        for i in range(1, n_iter + 1):

            dil = binary_dilation(segmentation, kernel)
            extra = (dil - segmentation)
            dilation_mask[np.where(extra == 1)] = i
            segmentation = dil.astype(int)

        dilation_mask[np.where(dil == 0)] = i

        return dilation_mask
