import numpy as np
from utils import find_center, find_extrema, create_circle, create_rotation_matrix, plot_convhull, plot_point_cloud, get_angle
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
        self.radii = [r1, r2, r3] #from proximal to distal
        self.geometry = geometry
        self.spacing = spacing

    def create_shape(self, num=20):
        """

        THIS IS OUTDATED CODE WITH IMPROPER GEOMETRY

        Creates the Cylinder shape
        :param num:
        :return:
        """

        # create the unit vector of the target and the entry points
        vhat = self.targ - self.ep

        rot1 = create_rotation_matrix(vhat,
                                      [1, 0, 0])

        target_coords_trans1 = self.targ - self.ep
        tumor_coords_rot_to_x_axis = rot1.apply(target_coords_trans1)
        tumor_center_rot_to_x_axis = rot1.apply(self.targ - self.ep)

        x_max, y_max, z_max = find_extrema(tumor_coords_rot_to_x_axis, 'max')

        x_offset = x_max - tumor_center_rot_to_x_axis[0]

        # generate points for the disc in the x plane

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
                                      vhat)
        # apply the rotation
        cyl_coords_rot = rot2.apply(cyl_coords_x_axis)

        # translate disc coordinates after rotation they are no longer centered around origin
        cyl_coords_rot_trans = cyl_coords_rot + self.ep

        # append all the vertexes into a surgical vertex list
        surgical_vertex = np.around(cyl_coords_rot_trans).astype(int)

        voxelized_cylinder = create_voxelized_path(surgical_vertex, cyl_coords_rot_trans, self.geometry, t='bi-cone')

        self.vtx = surgical_vertex
        self.voxel = voxelized_cylinder

    def create_shape2(self, num=20):
        """
        Create a cone in the X plane first, then rotate it to the correct space
        This does not have an error with the geometry of the shape. It will have the correct X-Y dimensions

        :param num: The number of points along the circumference that should be used to create
        :return:
        """

        # instantiate output array for data points
        cyl_coords_rot_scaled = []

        vhat = self.targ - self.ep

        for idx, radius in enumerate(self.radii):

            radius = radius/self.spacing[idx]  # correct the radius to compensate for voxel spacing

            path_length = np.linalg.norm(vhat)
            x_point = path_length * ((idx) / (len(self.radii) - 1))  # xpoint is a function of number of points of radii

            # create the y and z points of the circle
            y_circle, z_circle = create_circle(radius, radius, n=num)

            # create points for the x array
            x_circle = np.tile(x_point, len(y_circle))

            # combine all X,Y,Z points to make the circle
            cyl_coords_x_axis = list(zip(x_circle, y_circle, z_circle))
            center = np.array([x_point, 0, 0])

            # create the rotation between x axis and path vector
            rot2 = create_rotation_matrix([1, 0, 0], vhat)

            # rotate the points of the cylinder and center of cylinder by rot2
            cyl_coords_rot = rot2.apply(cyl_coords_x_axis) + self.ep
            center_rot = rot2.apply(center) + self.ep

            # for each point in cyl_coords_rot scale the radius based on the image spacing
            for p in cyl_coords_rot:
                A = p - center_rot
                p_new = center_rot + A / self.spacing
                cyl_coords_rot_scaled.append(p_new)

        # store as numpy array
        cyl_coords_rot_scaled = np.asarray(cyl_coords_rot_scaled)

        surgical_vertex = np.around(cyl_coords_rot_scaled).astype(int)
        voxelized_cylinder = create_voxelized_path(surgical_vertex, cyl_coords_rot_scaled, self.geometry, t='bi-cone')

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

    #TODO remove
    if np.any(np.isnan(non_voxelized_pts)):
       print('gotcha')
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

    ax = geometry

    voxelized_path_closed = binary_closing(voxelized_path,
                                           structure=np.ones((5, 5, 5))
                                           ).astype(int)
    return voxelized_path_closed

#im_pth =
#out_pth =


#EP = np.array([376, 300, 50])
#TARG = np.array([300, 215, 73])
#
#RADI = np.array([5, 2.5, 0.25])
#GEO = np.array([512, 512, 124])
#SPACING = np.array([0.3515625, 0.3515625, 0.49999723])
#cyl = Cylinder(EP, TARG, RADI[0], RADI[1], RADI[2], GEO, SPACING)
#cyl.create_shape2()
#