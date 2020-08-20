import numpy as np
from utils import find_center
from scipy.spatial import ConvexHull

def inhull(testpoints, hull, tol=0):
    """
    finds if point is in a 3d hull for simplex copied method from matlab function


    matlab function: inhull.mat
    author: John D'Errico
    email: woodchips@rochester.rr.com
    release date: 10/26/06

    :param testpoints: points to test with n elements in p dimensions size (n, p)
    :param hull: if no hull is passed then create the hull
    :param tol: tolerance?
    :return: outarray -- 1d array size n containing boolean for if the pt is contained in the hull
    """

    simplx = hull.simplices
    pts = hull.points

    p = pts.shape[1]
    n,c = testpoints.shape
    nt, c = simplx.shape

    if p != c:
        raise ValueError(" dimensions of vertex {} and testpoints {} not equal".format(p, c))

    # create vectorized cross products between simplex
    ab  = pts[simplx[:, 0], :] - pts[simplx[:, 1], :]
    ac  = pts[simplx[:, 0], :] - pts[simplx[:, 2], :]
    nrmls_non_calib = np.cross(ab, ac)
    degen = np.zeros((nt))

    # turn normals into unit vectors
    nrmllen = np.sqrt(np.sum(nrmls_non_calib**2, axis=1)).reshape(-1,1)
    nrmls = nrmls_non_calib / nrmllen

    center = find_center(hull.points[hull.vertices])

    #any point in the plane of each simplex in the convex hull
    any_pt = pts[simplx[np.logical_not(degen), 0], :]

    # make sure normals are pointing inwards
    dp = np.sum((np.tile(center, (nt, 1)) * nrmls), axis=1)
    k = dp < 0
    nrmls[k, :] = -1 * nrmls[k, :]

    # test if dot:((x-1), N) >= 0
    # if so for all faces of the hull, then x is inside the hull.
    aN = np.sum(nrmls * any_pt, axis=1).reshape(-1,1)

    # output this
    output = np.zeros(n)

    memblock = 100000
    blocks = np.max((1, np.floor(n/(memblock/nt)).astype(int)))
    aNr = np.tile(aN, len(np.arange(0, n, blocks)))
    for i in range(0, blocks):
        j = np.arange(i, n, blocks)
        if aNr.shape[0] != len(j):
            aNr = np.tile(aN, len(j))
        output[j] = ~np.all((np.matmul(nrmls, np.transpose(testpoints[j,:])) - aNr) >= -tol, axis=0)


    return output

def vectorized_in_hull(testpoints, hull):
    """

    calculate the dot product between the point and the normals of the convex hull. If the all dot products are less than
    zero then the point is within the hull.

    :param testpoints: nxc array of the points to test
    :param hull: convex hull object that we want to tsee if the test points go into
    :return:
       output: nx1 boolean array with True or False for if the testpoints are in the array
    """

    tol = 1e-6
    n, c = testpoints.shape

    nrmls = hull.equations[:,:-1]
    aN = hull.equations[:, -1]

    output = np.zeros(testpoints.shape[0]).astype(bool)

    memblock = 100000
    blocks = np.max((1, np.floor(n/(memblock)).astype(int)))
    aNr = np.tile(aN, (len(np.arange(0, n, blocks)),1)).T
    for i in range(0, blocks):
        j = np.arange(i, n, blocks)
        if aNr.shape[1] != len(j):
            aNr = np.tile(aN, (len(j), 1)).T
        output[j] = np.all((np.matmul(nrmls, np.transpose(testpoints[j,:])) + aNr <= tol), axis=0)

    return output

#mypts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
#                  [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10]])
#myhull = ConvexHull(mypts)
#print(vectorized_in_hull(np.array([[5, 5, 5], [7, 7, 7], [15, 15, 15], [10,11,10]]), myhull))


