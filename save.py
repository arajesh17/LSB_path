import nrrd
from costfunction import CostFunction
from surgicalpath import SurgicalPath



def save_path(center, target_coords, geometry, img_spacing, hdr, fname):
    """

    :param center:
    :param target_coords:
    :param geometry:
    :param img_spacing:
    :param hdr:
    :param fname:
    :return:
    """

    # create microscope
    micro_vtx, micro_pth = SurgicalPath.create_microscope(center, target_coords, geometry, img_spacing)

    #create cyl
    cyl_vtx, cyl_pth = SurgicalPath.create_bicone_cylinder(center, target_coords, geometry, img_spacing)

    nrrd.write(fname+'_cyl.nrrd', cyl_pth, hdr)
    nrrd.write(fname+'_micro.nrrd', micro_pth, hdr)