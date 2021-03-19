import numpy as np
from scipy.special import factorial, sph_harm, genlaguerre
import matplotlib.pyplot as plt
from dolfin import *
import pandas as pd
from mayavi import mlab
from skimage import measure

# build wave-function:
def wave_function(n, l, m, x, y, z, a0):

    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/r)
    phi = np.arctan(y/x)

    # replace possible NaNs wirh 0
    theta[np.argwhere(np.isnan(theta))] = 0
    phi[np.argwhere(np.isnan(phi))] = 0

    # Radial part
    R = np.sqrt((2/n/a0)**3*factorial(n-l-1)/(2*n*(factorial(n+l))**3)) * (2*r/n/a0)**l * np.exp(-r/n/a0) * genlaguerre(n-l-1,2*l+1)(2*r/n/a0)
    # spherical harmonics
    Y = sph_harm(m, l, phi, theta)
    # complete wavefunction
    wf = R * Y
    return wf


def main():

    d=0.1
    min=-10
    max=10
    X = np.arange(min,max,d)
    Y = np.arange(min,max,d)
    Z = np.arange(min,max,d)
    x, y, z = np.meshgrid(X, Y, Z)
    a0 =  1

#    mlab.figure()

    wf = np.abs(wave_function(2, 1, 0, x, y, z, a0))**2
    mlab.contour3d(wf, transparent=True)

    mlab.savefig('s-orbital.png')
    mlab.colorbar()
    mlab.outline()
    mlab.show()


if __name__ == "__main__":
    main()
