#!/usr/bin/python3.9


from dolfin import *

import numpy as np
import os


#verwenden atomare einheiten. hier werden alle in der schrodingergleicchung notigen naturkonstannten 1 und die langeneinheit wird der bohrradius

def system(i):

    d = i/100.0         #d in mesh units
    dh = d/2.0

    #define mesh and function space
    mesh = UnitCubeMesh(20, 20, 20)   #1 mesh unit = bohrradius = 0.529 Angstrom
    V = FunctionSpace(mesh, 'CG', 1)

    #   Potential for the Hydrogen Atom
    #       {  1/r      if r > 0
    #   V = {
    #       {   0       else

    formula = 'sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])>0 ?\
                      -(1/sqrt((x[0]-' + str(dh) + ')*(x[0]-' + str(dh) + ')+x[1]*x[1]+x[2]*x[2]) + 1/sqrt((x[0]+' + str(dh) + ')*(x[0]+' + str(dh) + ')+x[1]*x[1]+x[2]*x[2]) - 1.0/' + str(d) + ') : 0'



    pot = Expression(formula,degree=3)


    # Boundary 0 everywhere
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, 0, boundary)

    u = TrialFunction(V)
    v = TestFunction(V)

    # Assemble system
    a = (1/2*inner(grad(u), grad(v)) + pot*u*v)*dx

    # Define Matrices
    A = PETScMatrix()
    assemble(a, tensor=A)

    # Apply boundary on the system
    bc.apply(A)

    # Create eigensolver
    eigensolver = SLEPcEigenSolver(A)
    eigensolver.parameters["spectrum"] = "smallest magnitude"

    values = 1
    eigensolver.solve(values)

    u = Function(V)


    for j in range(values):
        E, E_c, R, R_c = eigensolver.get_eigenpair(j)

    print('E_0 = ', E)

    file = open("energy.dat","a")
    file.write(str(d))
    file.write("     ")
    file.write(str(E))
    file.write("\n")
    file.close

def main():
    for i in range(100, 300, 2):
        system(i)

if __name__ == "__main__":
    main()
