import numpy as np

"""
Code that could be used to randomly search for the optimal cranitomy search points"""

class CraniBounds(object):

    def __init__(self, testpoints):
        self.tp = testpoints

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        return any([all(x == p) for p in self.tp])


class RandomDisplacementBounds(object):

    def __init__(self, xmin, xmax, stepsize=40):

        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):

        while True:
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew
