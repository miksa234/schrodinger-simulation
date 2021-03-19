#!/usr/bin/python3.9

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv

import time

def indicator(V0, x, y, x0, x1, y0, y1):
    n = len(x)
    h = np.zeros(n*n)
    for i in range(n):
        for j in range(n):
            if x[i] >= x0 and x[i] <= x1:
                if y[j] >= y0 and y[j] <= y1:
                    h[i + j*n] = 1

    return V0*h

def hamiltonian(n, V, dx):
    dim, n = n, n*n

    d0 =  -2*(1/dx**2 + 1/dx**2) * np.ones(n) + V
    d1 = np.array([1/dx**2 if i%dim != 0 else 0 for i in range(1, n)])
    dn = 1/dx**2 * np.ones(n-dim)

    return d0, d1, dn

def left_side(n, V, dx, dt):
    dim, n = n, n*n

    d0, d1, dn = hamiltonian(dim, V, dx)

    d_0 = -dt/2 * d0 + 1j
    d_1 = -dt/2 * d1
    d_n = -dt/2 * dn

    return sparse.diags([d_n, d_1, d_0, d_1, d_n], [-dim, -1, 0, 1, dim], format='csc')


def right_side(n, V, dx, dt):
    dim, n = n, n*n

    d0, d1, dn = hamiltonian(dim, V, dx)

    d_0 = dt/2 * d0 + 1j
    d_1 = dt/2 * d1
    d_n = dt/2 * dn

    return sparse.diags([d_n, d_1, d_0, d_1, d_n], [-dim, -1, 0, 1, dim], format='csc')


def wave_packet(x, y, dx, x0, y0, k0):
    return 1/(2*dx**2*np.pi)**(1/2)  *\
            np.exp(-((x-x0)/(2*dx)) ** 2) *\
            np.exp(-((y-y0)/(2*dx)) ** 2) *\
            np.exp(1.j * (k0*x))

def mysolver(A, M, u0, dt, steps):
    U = np.array([u0] + [np.zeros(u0.size) for i in range(steps)])
    for i in range(steps):
        U[i+1] = spsolve(A, M.dot(U[i]))
    return U

def norm(u, x):
    n = len(x)
    u = np.sqrt(np.real(u)**2 + np.imag(u)**2)**2
    u = u.reshape(n, n)
    return np.trapz(np.trapz(u, x), x)

def convert(U, x, y):
    """normalization of each solution"""
    n = len(x)
    Unew = []
    for u in U:
        unew = abs(u)**2/norm(u, x)
        Unew.append(unew.reshape(n, n))
    return Unew

