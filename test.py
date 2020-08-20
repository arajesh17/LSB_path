import numpy as np
from loaddata import LoadData
from utils import binarize_segmentation
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy import optimize
from surgicalpath import SurgicalPath
from utils import plot_convhull, plot_point_cloud, find_extrema
from scipy.spatial import ConvexHull, Delaunay
import nrrd


img_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\601 AX 3D B FFE IACs_1.nrrd'
seg_pth = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\Pt_6_Segmentation-label.nrrd'

lup_tbl = 'C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\Pt_6_Segmentation-label_ColorTable.ctbl'

im, hdr = nrrd.read(img_pth)

def convert_ijk_to_RAS(hdr, pt):

    spacing = np.hstack((hdr['space directions'].T, hdr['space origin'].reshape(-1, 1)))
    ijk_to_lps = np.vstack((spacing, [0, 0, 0, 1]))
    lps_to_ras = np.diag([-1, -1, 1, 1])
    ijk_to_ras = np.matmul(lps_to_ras, ijk_to_lps)
    print('hi')

def convert_ijk_to_RAS(hdr, pt):

    # turn into the [x,y,z, 1] format
    if pt.shape == (3,):
        pt = np.concatenate(pt, [1]).reshape(-1, 1)

    if pt.shape != (1, 4):
        return ValueError('Point shape is {} but needs to be (4,1))')

    spacing = np.hstack((hdr['space directions'].T, hdr['space origin'].reshape(-1, 1)))
    ijk_to_lps = np.vstack((spacing, [0, 0, 0, 1]))
    lps_to_ras = np.diag([-1, -1, 1, 1])
    ijk_to_ras = np.matmul(lps_to_ras, ijk_to_lps)
    np.matmul(ijk_to_ras, pt)

def convert_RAS_to_ijk(hdr, pt):

    # turn into the [x,y,z, 1] format
    if pt.shape == (3,):
        pt = np.concatenate(pt, [1]).reshape(-1,1)

    if pt.shape != (1, 4):
        return ValueError('Point shape is {} but needs to be (4,1))')

    spacing = np.hstack((hdr['space directions'].T, hdr['space origin'].reshape(-1, 1)))
    ijk_to_lps = np.vstack((spacing, [0, 0, 0, 1]))
    lps_to_ras = np.diag([-1, -1, 1, 1])
    ijk_to_ras = np.matmul(lps_to_ras, ijk_to_lps)
    ras_to_ijk = np.invert(ijk_to_ras)
    np.matmul(ras_to_ijk, pt)



def find_edge_points(target):
    """
    Finds the boundary points of the target by eroding it by three iterations
    Then it finds the extrema of x,y,z points

    :param target:
    :return:
    """

    # erode 3 times
    struct = generate_binary_structure(3,1)
    eroded = binary_erosion(target, structure= struct, iterations=3).astype(int)
    boundary = eroded - binary_erosion(eroded, structure=struct).astype(int)

    # key points:
    key_points = []
    coords = np.array(np.where(boundary == 1)).T

    low = find_extrema(coords, 'min')
    high = find_extrema(coords, 'max')

    for d in [[0,1,2], [2, 0, 1], [1, 2, 0]]:

        # find low
        low_dim1 = coords[coords[:, d[0]] == low[d[0]]]

        max_dim2 = low_dim1[low_dim1[:, d[1]] == np.max(low_dim1[:, d[1]])]
        max_dim3 = max_dim2[max_dim2[:, d[2]] == np.max(max_dim2[:, d[2]])]
        key_points.append(max_dim3)
        min_dim3 = max_dim2[max_dim2[:, d[2]] == np.min(max_dim2[:, d[2]])]
        key_points.append(min_dim3)

        min_dim2 = low_dim1[low_dim1[:, d[1]] == np.min(low_dim1[:, d[1]])]
        max_dim3 = min_dim2[min_dim2[:, d[2]] == np.max(min_dim2[:, d[2]])]
        key_points.append(max_dim3)
        min_dim3 = min_dim2[min_dim2[:, d[2]] == np.min(min_dim2[:, d[2]])]
        key_points.append(min_dim3)

        # find high
        high_dim1 = coords[coords[:, d[0]] == high[d[0]]]

        max_dim2 = high_dim1[high_dim1[:, d[1]] == np.max(high_dim1[:, d[1]])]
        max_dim3 = max_dim2[max_dim2[:, d[2]] == np.max(max_dim2[:, d[2]])]
        key_points.append(max_dim3)
        min_dim3 = max_dim2[max_dim2[:, d[2]] == np.min(max_dim2[:, d[2]])]
        key_points.append(min_dim3)

        min_dim2 = high_dim1[high_dim1[:, d[1]] == np.min(high_dim1[:, d[1]])]
        max_dim3 = min_dim2[min_dim2[:, d[2]] == np.max(min_dim2[:, d[2]])]
        key_points.append(max_dim3)
        min_dim3 = min_dim2[min_dim2[:, d[2]] == np.min(min_dim2[:, d[2]])]
        key_points.append(min_dim3)

    plot = False
    if plot == True:
        pts = np.vstack(tuple(x for x in key_points))
        plot_point_cloud(np.where(boundary == 1), pts.T)

    return key_points


'''
# create affine
aff = hdr["space dimensions"]
aff = np.hstack((aff, hdr['space origin'].reshape(-1,1)))
aff = np.vstack((aff, [[0, 0, 0, 1]]))
'''

# make the coordinates

#possible_pts = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4]])
#
#def cf(x):
#    if any([all(x == p) for p in possible_pts]):
#        return x[0] ** 2 + x[1] * 4 * x[0] ** 3 + x[2] ** 4
#    else:
#        return np.inf
#
#res = optimize.basinhopping(cf, [1,1,1], niter=100)

#pth = "C:\\Users\\anand\\OneDrive - UW\\LSB_cohort\\pt_6\\Pt_6_Segmentation-label.nrrd"
#seg, hdr = nrrd.read(pth)
#
#print("hello")
#
#crani_coords = np.array([[10,100,100]])
#tumor_coords = np.array([[15,100,90]])

pts = np.array([
   np.array( [0,0,0]),
   np.array( [10,0,0]),
   np.array( [10,10,0]),
   np.array( [10,10,10]),
   np.array( [0,10,10]),
   np.array( [0,0,10]),
   np.array( [0,10,0]),
   np.array( [10,0,10]),
    ]
)

pts2 = np.array([
    np.array([0,10,10]),
    np.array([10,0,10]),
    np.array([10,10,10]),
    np.array([0,0,10]),
    np.array([10,0,0]),
    np.array([0,0,0]),
    np.array([10,10,0]),
    np.array([0,10,0]),
    np.array([5,10,10]),
    np.array([5,10,0]),
    np.array([5,0,10])
    ]
)

def check_tri_intersection(param):
    """
    Chekc intersection foundf rom answer in stack overflow
    https://stackoverflow.com/questions/55220355/how-to-detect-whether-two-segmentin-3d-spaceintersect

    """

    # set tolerance for machine imprecision
    tol = 1e-6
    value = False

    while value == False:

        order = [0,1,2,3]
        random.shuffle(order)
        i1, i2, i3, i4 = order

        P1 = param[i1]
        P2 = param[i2]
        Pd = P2-P1
        Pm = np.cross(P1, P2)

        stack = np.vstack((P1, P2))
        bnds = np.array(list(zip(stack.min(axis=0), stack.max(axis=0))))

        Q1 = param[i3]
        Q2 = param[i4]
        Qd = Q2-Q1
        Qm = np.cross(Q1, Q2)

        if abs(np.dot(Pd, Qm) + np.dot(Qd, Pm)) > tol:
            # if they are not co-planar than they cannot intersect
            continue

        # check to see if two lines are parallel by finding their cross product
        if np.linalg.norm(np.cross(Pd, Qd)) < tol:
            continue

        x = np.cross(Pm, Qm) / np.dot(Pd, Qm)

        # find if intersection point is within bounds
        if all([bnds[k, 0] < x[k] < bnds[k, 1] for k in range(len(x))]):
            a = P1
            d = P2
            b = Q1
            c = Q2
            return a,b,c,d

#parameters = np.array([[398, 378,  57],
#                  [363, 474,  52],
#                  [331, 420,   1],
#                  [381, 367,   2]])
#
#vals = sort_corners(parameters)
#print(vals)


# [420, 193,  30],
# [419, 235,  33]])
#
#from scipy.spatial import ConvexHull
#hully = ConvexHull(pts)
#D = Delaunay(pts)
#test_points = create_coordinate_window(pts)
#
#outpt = []
#for t in test_points:
#    outpt.append(in_hull2(t, D))
#
#def flood_fill_hull(image):
#    points = np.transpose(np.where(image))
#    hull = ConvexHull(points)
#    deln = Delaunay(points[hull.vertices])
#    idx = np.stack(np.indices(image.shape), axis = -1)
#    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
#    out_img = np.zeros(image.shape)
#    out_img[out_idx] = 1
#    return out_img, hull
#
#points = tuple(np.rint(10 * np.random.randn(3,100)).astype(int) + 50)
#image = np.zeros((100,)*3)
#image[points] = 1
#
#
#out, h = flood_fill_hull(image)



'''
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import numpy as np

hull = ConvexHull( np.array([[303, 196, -7], [304, 188, -5], [304, 180, 0], [304, 175, 6], [303, 171,  14], [302, 171,  23], [301, 174,  31], [299, 178,  38], [298, 185,  43], [296, 194,  45], [295, 202,  45], [295,  210,  42], [294, 217,  36], [295, 221,  29], [296, 223,  20], [297, 222,  12], [298, 218,   4]]))
hull_pth = Path([hull.vertices])

z_pts =  np.ones((50, 30, 10))
pts = np.array(list(zip(z_pts[0], z_pts[1], z_pts[2])))

output = np.zeros((200, 200, 200))
for pt in pts:
    if hull_pth.contains_path(pt):
        output[pt[0], pt[1], pt[2]] = 1
'''
