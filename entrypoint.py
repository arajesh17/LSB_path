import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_dilation
from coordinates import convert_RAS_to_ijk

class EntryPoint:

    def __init__(self, fcsv, hdr):
        """
        Class to wrap different methods to find the entry points of the input craniotomy

        Parameters
        ----------
        fcsv: str
            the filename of the fcsv file with the fiducial information
        hdr: str
            the filename of the header for the image
        """
        self.fcsv = fcsv
        self.hdr = hdr

    def skull_strip(self):
        """ Method to detect where the skull superficial border is:
        dilate the tisse segmentation by one voxel then find the intersection fo that with the bone
        This will define the superficial boundary of the cortical bone
        Only if the tissue is segmented so that it

        :returns
            entrypoints: the nx3 array of the potential entry points for starting cylinder approaches
            each entrypoint is derived from the stripped skull
        """

        # dilate tissue
        dil_tiss = binary_dilation(self.tissue, structure=np.ones((3,3,3))) # make this
        u_tiss_bone = np.logical_and(dil_tiss, self.skull).astype(int)

        # create islands and remove the small ones :)
        labeled_array, num_features = label(u_tiss_bone)
        for i in range(num_features):
            if len(np.where(labeled_array == i)[0]) < 100:
                u_tiss_bone[np.where(labeled_array == i)] = 0

        self.strip = u_tiss_bone

        sort = sorted(np.array(np.where(u_tiss_bone == 1)).T, key=lambda k: [k[0], k[1], k[2]])
        entrypoints = np.array(sort)

        return entrypoints

    def from_fiducials(self):
        """
        computes boundary points of craniotomy from the fiducial marks created in slicer
        Returns
        -------
        MCF_ijk: ndarray
            MCF fiducial coordinates in ijk coordinate system
        RS_ijk: ndarray
            RS fiducial coordinates in ijk coordinate system
        """

        # load the data from the the fiducials
        f = pd.read_csv(self.fcsv, header=2)

        # MCF
        MCF_RAS = f.loc[f['desc'] == 'MCF'][['x', 'y', 'z']].values
        MCF_ijk = np.array([convert_RAS_to_ijk(self.hdr, x) for x in MCF_RAS])

        # RS
        RS_RAS = f.loc[f['desc'] == 'RS'][['x', 'y', 'z']].values
        RS_ijk = np.array([convert_RAS_to_ijk(self.hdr, x) for x in RS_RAS])

        #debugging to set an negative values to zero in case fiducials are placed outside of range
        MCF_ijk[np.where(MCF_ijk < 0)] = 0
        RS_ijk[np.where(RS_ijk < 0)] = 0

        return MCF_ijk, RS_ijk
