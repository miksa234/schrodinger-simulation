#!/usr/bin/python3.9

from dolfin import *
import numpy as np
import os

def main():
    mesh = UnitCubeMesh(25, 25, 25)
    V = FunctionSpace(mesh, 'CG', 1)

    m_e = 9.10e-31; m_k = 1.67e-27
    hbar = 1.05e-34; k = 8.98e9
    mu = m_e*m_k/(m_e + m_k)
    e = 1.60e-19

    #   Potential for the Hydrogen Atom
    #       {  1/r      if r > 0
    #   V = {
    #       {   0       else
    pot_ = Expression('sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])>0 ?\
                     -1/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) : 0',\
                     degree=4)
    pot = interpolate(pot_, V)


    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(0.), boundary)

    u = TrialFunction(V)
    v = TestFunction(V)

    # Assemble system
    a = hbar/(2*mu)*inner(grad(u), grad(v))*dx + e**2/k*pot*u*v*dx
    m = u*v*dx

    # Define Matrices
    A = PETScMatrix()
    M = PETScMatrix()

    assemble(a, tensor=A)
    assemble(m, tensor=M)

    # Apply boundary on the system
    bc.apply(A)
    bc.apply(M)

    # Create eigensolver
    eigensolver = SLEPcEigenSolver(A, M)
    eigensolver.parameters["spectrum"] = "smallest magnitude"

    values = 30
    eigensolver.solve(values)

    u = Function(V)


    if os.path.exists('./meshes') != True:
        os.mkdir('./meshes')

    f = File('./meshes/orbitals.pvd')

    for i in range(values):
        E, E_c, R, R_c = eigensolver.get_eigenpair(i)

        u.vector()[:] = np.sqrt(R*R + R_c*R_c)

        f << (u, i)

if __name__ == "__main__":
    main()
