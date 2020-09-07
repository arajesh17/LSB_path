import numpy as np
from utils import find_center, find_extrema, create_circle, create_rotation_matrix, plot_convhull, plot_point_cloud
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage.morphology import binary_closing
from weights import microscope_dict
from inhull import vectorized_in_hull
import nrrd

class Cylinder(object):
    """ Cylinder"""

    def __init__(self, entrypoint, target, r1, r2, r3, geometry, spacing):

        self.ep = entrypoint
        self.targ = target
        self.radii = [r1, r2, r3]
        self.geometry = geometry
        self.spacing = spacing

    def create_shape(self, num=20):
        """
        Creates the Cylinder shape
        :param num:
        :return:
        """

        target_coords_trans1 = self.targ - self.ep
        rot1 = create_rotation_matrix(self.targ - self.ep,
                                      [1, 0, 0])

        tumor_coords_rot_to_x_axis = rot1.apply(target_coords_trans1)
        tumor_center_rot_to_x_axis = rot1.apply(self.targ - self.ep)

        x_max, y_max, z_max = find_extrema(tumor_coords_rot_to_x_axis, 'max')

        x_offset = x_max - tumor_center_rot_to_x_axis[0]

        # generate points for the disc in the x plane #todo make the radius a passable parameter
        y_circle, z_circle = create_circle(self.radii[0] / self.spacing[1], self.radii[0] / self.spacing[2],
                                           n=num)  # calibrate radius based off of spacing
        y_circle2, z_circle2 = create_circle(self.radii[1] / self.spacing[1], self.radii[1] / self.spacing[2],
                                             n=num)  # calibrate radius based off of spacing
        y_circle3, z_circle3 = create_circle(self.radii[2] / self.spacing[1], self.radii[2] / self.spacing[2],
                                             n=num)  # calibrate radius based off of  spacing
        path_length = np.linalg.norm(self.targ - self.ep) + x_offset
        midpoint = path_length / 2
        x_circle = np.tile(0, len(y_circle))
        x_circle2 = np.tile(midpoint, len(y_circle2))
        x_circle3 = np.tile(path_length, len(y_circle3))
        circle1 = list(zip(x_circle, y_circle, z_circle))
        circle2 = list(zip(x_circle2, y_circle2, z_circle2))
        circle3 = list(zip(x_circle3, y_circle3, z_circle3))
        cyl_coords_x_axis = np.vstack((circle1, circle2, circle3))

        # create the rotation between x axis and path vector
        rot2 = create_rotation_matrix([1, 0, 0],
                                      self.targ - self.ep)

        # apply the rotation
        cyl_coords_rot = rot2.apply(cyl_coords_x_axis)

        # translate disc coordinates after rotation they are no longer centered around origin
        cyl_coords_rot_trans = cyl_coords_rot + self.ep

        # append all the vertexes into a surgical vertex list
        surgical_vertex = np.around(np.array(cyl_coords_rot_trans)).astype(int)

        voxelized_cylinder = create_voxelized_path(surgical_vertex, cyl_coords_rot_trans, self.geometry, t='bi-cone')

        self.vtx = surgical_vertex
        self.voxel = voxelized_cylinder

    def save(self, hdr, fname):
        """
        saves the cylinder
        :param: hdr The header of the nrrd file to save
        :param: fname the filename of the output
        :return:
        """
        nrrd.write(fname, self.voxel, fname, hdr)
        print('saved file at {}'.format(fname))


class SurgicalPath():
    """
    create the vertexes of a shape with craniotomy and the
    disc with radius of the tumor target
    """


    @staticmethod
    def create_microscope(crani_center, target_coords, geometry, img_spacing, micro_dict=microscope_dict):
        """
        Creates a computed microscope path2

        :param crani_center:
        :param target_coords:
        :param geometry:
        :param img_spacing:
        :return:
        """

        if len(crani_center.shape) > 1:
            crani_center = find_center(crani_center)
        target_center = find_center(target_coords)

        target_coords_trans1 = target_coords - crani_center
        rot1 = create_rotation_matrix(target_center - crani_center,
                                      [1, 0, 0])

        target_coords_rot_to_x_axis = rot1.apply(target_coords_trans1)
        target_center_rot_to_x_axis = rot1.apply(target_center - crani_center)

        x_min, y_min, z_min = find_extrema(target_coords_rot_to_x_axis, 'min')
        x_max, y_max, z_max = find_extrema(target_coords_rot_to_x_axis, 'max')

        # find the widest point of the target coordinates rotated on the x axis
        y_radius = (y_max - y_min) / 2
        z_radius = (z_max - z_min) / 2

        if y_radius * img_spacing[1] < micro_dict["Min Distal Radius"]:
            y_radius = micro_dict["Min Distal Radius"] / img_spacing[1]
        if z_radius * img_spacing[2] < micro_dict["Min Distal Radius"]:
            z_radius = micro_dict["Min Distal Radius"] / img_spacing[2]

        # generate points for the disc in the x plane
        y_circle, z_circle = create_circle(y_radius, z_radius, n=20)
        path_length = np.linalg.norm(target_center - crani_center)
        x_offset = x_max - target_center_rot_to_x_axis[0]
        x_circle = np.tile(path_length, len(y_circle)) + x_offset

        ratio = micro_dict["Radius"]/micro_dict["Focal Length"]
        y_micro, z_micro = create_circle(path_length * ratio / img_spacing[1],
                                         path_length * ratio / img_spacing[2],
                                         n=20)
        x_micro = np.tile(0, len(y_micro))
        tumor_x_axis = np.array(list(zip(x_circle, y_circle, z_circle)))
        micro_x_axis = np.array(list(zip(x_micro, y_micro, z_micro)))
        disc_coords_x_axis = np.vstack((tumor_x_axis, micro_x_axis))

        # create the rotation between x axis and path vector
        rot2 = create_rotation_matrix([1, 0, 0],
                                      target_center - crani_center)  # TODO make a vector

        # apply the rotation
        disc_coords_rot = rot2.apply(disc_coords_x_axis)

        # translate disc coordinates after rotation they are no longer centered around origin
        surgical_vertex = disc_coords_rot + crani_center

        # round values
        surgical_vertex_round = np.around(np.array(surgical_vertex)).astype(int)

        voxelized_disc_w_crani = create_voxelized_path(surgical_vertex_round, surgical_vertex, geometry, t='crani_disc')

        return surgical_vertex, voxelized_disc_w_crani

    @staticmethod
    def create_disc_with_crani( crani_coords, target_coords, geometry, img_spacing):

        target_center = find_center(target_coords)
        crani_center = find_center(crani_coords)

        target_coords_trans1 = target_coords - crani_center
        rot1 = create_rotation_matrix(target_center - crani_center,
                                                   [1, 0, 0])

        target_coords_rot_to_x_axis = rot1.apply(target_coords_trans1)
        target_center_rot_to_x_axis = rot1.apply(target_center - crani_center)

        x_min, y_min, z_min = find_extrema(target_coords_rot_to_x_axis, 'min')
        x_max, y_max, z_max = find_extrema(target_coords_rot_to_x_axis, 'max')


        # find the widest point of the target coordinates rotated on the x axis
        y_radius = (y_max - y_min)/2
        z_radius = (z_max - z_min)/2

        # generate points for the disc in the x plane
        y_circle, z_circle = create_circle(y_radius, z_radius, n=20)
        path_length = np.linalg.norm(target_center - crani_center)
        x_offset = x_max-target_center_rot_to_x_axis[0]
        x_circle = np.tile(path_length, len(y_circle)) + x_offset
        disc_coords_x_axis = np.array(list(zip(x_circle, y_circle, z_circle)))

        # create the rotation between x axis and path vector
        rot2 = create_rotation_matrix([1, 0, 0],
                                      target_center - crani_center) #TODO make a vector

        # apply the rotation
        disc_coords_rot = rot2.apply(disc_coords_x_axis)

        # translate disc coordinates after rotation they are no longer centered around origin
        disc_coords_rot_trans = disc_coords_rot + crani_center

        # append all the vertexes into a surgical vertex list
        surgical_vertex = []
        for i in disc_coords_rot_trans:
            surgical_vertex.append(i)
        for i in crani_coords:
            surgical_vertex.append(i)
        surgical_vertex_round = np.around(np.array(surgical_vertex)).astype(int)

        voxelized_disc_w_crani = create_voxelized_path(surgical_vertex_round, surgical_vertex, geometry, t='crani_disc')

        return surgical_vertex, voxelized_disc_w_crani

    @staticmethod
    def create_bicone_cylinder(crani_coords, target_coords, r1, r2, r3, geometry, spacing, n=20):
        """
        create a biconical cylinder similar to the anterior skull base. create it in the x plane then rotate it to the
        the correct plane of the crani_coords, target_coords vector


        :param proximal_center:
        :param distal_center:
        :param geometry: geometry of the image
        :param r1: radius  of the proximal circle of bicone
        :param r2: radius of the midpoitn circle of the bicone
        :param r3: radius of the distal circle of the bicone
        :param spacing: image spacing of the slices of each axis in mm i.e. 1mm x 2mm x 3mm [1.0, 2.0, 3.0]
        :param n: number of vertices per circle
        :returns vertex
        :returns path

        """

        proximal_center = find_center(crani_coords)
        target_center = find_center(target_coords)

        target_coords_trans1 = target_coords - proximal_center
        rot1 = create_rotation_matrix(target_center - proximal_center,
                                                   [1, 0, 0])

        tumor_coords_rot_to_x_axis = rot1.apply(target_coords_trans1)
        tumor_center_rot_to_x_axis = rot1.apply(target_center - proximal_center)

        x_max, y_max, z_max = find_extrema(tumor_coords_rot_to_x_axis, 'max')

        x_offset = x_max - tumor_center_rot_to_x_axis[0]

        # generate points for the disc in the x plane #todo check logic here
        y_circle, z_circle = create_circle(r1/spacing[1], r1/spacing[2], n=20) #calibrate radius based off of spacing
        y_circle2, z_circle2 = create_circle(r2/spacing[1], r2/spacing[2], n=20) #calibrate radius based off of spacing
        y_circle3, z_circle3 = create_circle(r3/spacing[1], r3/spacing[2], n=20) #calibrate radius based off of  spacing
        path_length = np.linalg.norm(target_center - proximal_center) + x_offset
        midpoint = path_length / 2
        x_circle = np.tile(0, len(y_circle))
        x_circle2 = np.tile(midpoint, len(y_circle2))
        x_circle3 = np.tile(path_length, len(y_circle3))
        circle1 = list(zip(x_circle, y_circle, z_circle))
        circle2 = list(zip(x_circle2, y_circle2, z_circle2))
        circle3 = list(zip(x_circle3, y_circle3, z_circle3))
        cyl_coords_x_axis = np.vstack((circle1, circle2, circle3))

        # create the rotation between x axis and path vector
        rot2 = create_rotation_matrix([1, 0, 0],
                                      target_center - proximal_center) #todo make a vector

        # apply the rotation
        cyl_coords_rot = rot2.apply(cyl_coords_x_axis)

        # translate disc coordinates after rotation they are no longer centered around origin
        cyl_coords_rot_trans = cyl_coords_rot + proximal_center

        # append all the vertexes into a surgical vertex list
        surgical_vertex = np.around(np.array(cyl_coords_rot_trans)).astype(int)

        voxelized_cylinder = create_voxelized_path(surgical_vertex, cyl_coords_rot_trans, geometry, t='bi-cone')

        return surgical_vertex, voxelized_cylinder


def create_coordinate_window(pts, shape):
    """
    :param pts: nxp array of points in shape (n, p)
    :return:  coords :array in shape (x_range*y_range*z_range , p)
    #TODO there is a bug with this function: if the pts are negative -- then it will return negative values and these
    will be plotted at the limits of the shape of the output array
    """

    def lowlim(floor):
        if floor < 0:
            return 0
        else:
            return floor

    def highlim(roof, position):
        if roof > shape[position]:
            return shape[position]
        else:
            return roof

    _min = find_extrema(pts, 'min')
    _max = find_extrema(pts, 'max')

    xx, yy, zz = np.meshgrid(np.arange(lowlim(_min[0]), highlim(_max[0], 0) + 1),
                             np.arange(lowlim(_min[1]), highlim(_max[1], 1) + 1),
                             np.arange(lowlim(_min[2]), highlim(_max[2], 2) + 1))

    coords = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).T

    return coords


def create_voxelized_path(pts, non_voxelized_pts, geometry, t=''):
    """
    Creates a voxelized form with the inputted vertices
    """
    convhull = ConvexHull(non_voxelized_pts)
    delan = Delaunay(non_voxelized_pts)
    print('number of vertices for the delaunay shape {}'.format(delan.vertices.shape))
    testpoints = create_coordinate_window(pts, geometry)

    bool_out = vectorized_in_hull(testpoints, convhull)

    voxelized_path = np.zeros(geometry)
    tp = testpoints[bool_out]

    # remove the points which fall outside of the geometry space of the image to not create out of range error
    idx = np.where((tp[:, 0] >= 0) & (tp[:, 0] < geometry[0]) &
                   (tp[:, 1] >= 0) & (tp[:, 1] < geometry[1]) &
                   (tp[:, 2] >= 0) & (tp[:, 2] < geometry[2]))
    within = tp[idx].T
    voxelized_path[within[0], within[1], within[2]] = True

    #plot_convhull(convhull, np.where(voxelized_path==True), plt_title=t)
    # plot_point_cloud(np.where(voxelized_path == True))
    #plot_convhull(delan)

    voxelized_path_closed = binary_closing(voxelized_path,
                                           structure=np.ones((5, 5, 5))
                                           ).astype(int)
    return voxelized_path_closed
