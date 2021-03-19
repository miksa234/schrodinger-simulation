#!/usr/bin/python3.9

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import os

from system import *
from plotter import *

import time

"""
    Solves the 2D time dependent Schrodinger Eq. on a square mesh.
    Initial condition is a Gaussian wave packet.
    The Potential is in form of the double/single slit

                    H * Psi = Psi_t

    FD approximation of the Laplace Operator,
    Crank Nicolson time discretization and
    hbar = m = 1 makes the system equal to the following

        (i  - dt/2 * H) * Psi_{n+1} = (dt/2 * H + i) * Psi_n
"""

def main():
    dx = 0.7; dt = 0.2
    n = 100; steps = 100
    xmin = 0; xmax = 10

    x = np.linspace(xmin, xmax, n)
    X, Y = np.meshgrid(x, x)

    x0 = 8; y0 = 5
    k0 = -50; V0 = 100

    # Initial Condition
    psi0 = wave_packet(X, Y, dx, x0, y0, k0)
    psi0 = (psi0/norm(psi0, x)).T.reshape(n*n)

    # Potential
    pos = 5; thickness = 0.07
    xs1 = 4.6; xs2 = 5.4;

    h0 = indicator(V0, x, x, xmin, xs1, pos, pos+thickness)
    h1 = indicator(V0, x, x, xs2, xmax, pos, pos+thickness)
    h2 = indicator(V0, x, x, 4.8, 5.2, pos, pos+thickness)
    V = h0 + h1 + h2

    #spalt = 0.2
    #for i in range(1, int((xs2-xs1)/spalt)):
    #    if i%3 != 0:
    #        V += indicator(V0, x, x, xs1+spalt*i, xs1+spalt*(i+1), pos, pos+thickness)


    # Assemble system
    A = left_side(n, V, dx, dt)
    M = right_side(n, V, dx, dt)

    start = time.time()
    Psi = mysolver(A, M, psi0, dt, steps)
    timing = time.time() - start

    print(f'Time for the calculation: {round(timing, 2)}')

    U = convert(Psi, x, x)

    plotter(U, V, dt, X, Y, 'single_slit')

if __name__ == "__main__":
    main()

