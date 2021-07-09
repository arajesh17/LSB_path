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
        Create a cone in the X plane first, then rotate it to the correct space
        This does not have an error with the geometry of the shape. It will have the correct X-Y dimensions

        Parameters
        ----------
        num: int
            The number of points along the circumference that should be used to create

        Returns
        -------

        """

        # instantiate output array for data points
        cyl_coords_rot_scaled = []

        vhat = self.targ - self.ep

        for idx, radius in enumerate(self.radii):

            radius = radius/self.spacing[idx]  # correct the radius to compensate for voxel spacing #TODO bug

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

def create_coordinate_window(pt_cloud_coords, shape):
    """
    returns a set of coordinates bound by the points

    Parameters
    ----------
    pt_cloud_coords: array [n, p]
        array of points with n as the dimension of the points and p as the number of points
    shape: array
        array of shape of point cloud space in n dimensions

    Returns
    -------

    """
    # round coordinates to integers
    pts = np.around(pt_cloud_coords).astype(int)

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


def create_voxelized_path(pts, geometry):
    """
    takes the points of a shape in cartesian space and converts them into a convex hull, then converts the convex hull
    into a point cloud in voxel space. then applies a binary closing to voxelized shape to close the voxelized shape.

    Parameters
    ----------
    pts: array  [N, P]
        list of vertices of the shape in form NxP where N is number of axes of coordinates and P is number of vertices
    geometry: array [N]
        geometry of the point cloud space array

    Returns
    -------
    voxelized_path_closed: array
        point cloud of voxels for inputted vertices

    """
    #TODO go sequetially to see the timing of each of the steps of this function to optimize for speed

    if np.any(np.isnan(pts)):
       raise ValueError("a point passed to be voxelized has a null value")

    convhull = ConvexHull(pts)
    testpoints = create_coordinate_window(pts, geometry)

    # find the voxels tht fall within the convex hull
    bool_out = vectorized_in_hull(testpoints, convhull)

    voxelized_path = np.zeros(geometry)
    tp = testpoints[bool_out]

    # remove the points which fall outside of the geometry space of the image to not create out of range error
    idx = np.where((tp[:, 0] >= 0) & (tp[:, 0] < geometry[0]) &
                   (tp[:, 1] >= 0) & (tp[:, 1] < geometry[1]) &
                   (tp[:, 2] >= 0) & (tp[:, 2] < geometry[2]))
    within = tp[idx].T
    voxelized_path[within[0], within[1], within[2]] = True

    # close shape to fill any holes
    voxelized_path_closed = binary_closing(voxelized_path,
                                           structure=np.ones((5, 5, 5))
                                           ).astype(int)
    return voxelized_path_closed
