#!/usr/bin/python3.9

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, hermite, eval_laguerre, genlaguerre
import matplotlib.patches as mpatches

def harmonic_oscillator(xmin, xmax, mesh, V):

    xmin=xmin
    xmax=xmax
    mesh=mesh
    V=V

    def boundary(x, on_bnd):
        return on_bnd

    bc = DirichletBC(V, Constant(0.), boundary)

    u = TrialFunction(V)
    v = TestFunction(V)

    pot_ = Expression('0.5*pow(x[0], 2)', degree=2)
    pot = interpolate(pot_, V)

    a = 1/2*inner(grad(u), grad(v))*dx + pot*u*v*dx
    m = u*v*dx

    A = PETScMatrix()
    M = PETScMatrix()

    assemble(a, tensor=A)
    assemble(m, tensor=M)

    bc.apply(A)

    # create eigensolver
    eigensolver = SLEPcEigenSolver(A, M)
    eigensolver.parameters['spectrum'] = 'smallest magnitude'

    # solve for eigenvalues
    values = 10
    eigensolver.solve(values)

    u = Function(V)

    # plot
    fig = plt.figure(figsize=[10, 10])
    En = []
    for i in range(0, values+1):
        E, E_c, R, R_c = eigensolver.get_eigenpair(i)
        En.append(E)
        x = np.linspace(xmin, xmax, len(R))

        I = np.trapz(R*R, x)
  #      print(I)

        #plot eigenfunction
        plt.plot(x, 20*R*R+E, c='blue')
        plt.plot(x, R_c+E, lw=0.5, c='black')
        plt.annotate(f'$E_{ { i } } = {round(En[i], 3)}$', (xmax-0.1, 0.05+E), fontsize=14)
        plt.annotate(f'$|\Psi _{ { i } }|²$', (xmin, 0.1+E), fontsize=14)

    plt.plot(x, 1/2*x**2, c='red')
    plt.ylim(0, max(En)+0.5)
    plt.title('Harmonic Oscillator $V(x) =\\frac{1}{2}x^2 $', fontsize=20)
    plt.xticks([])
    plt.yticks([])



    # plot analytical solutions

    def harm_osc(n, x):
        return 1/sqrt(sqrt(np.pi)*(2**n)*factorial(n)) * hermite(n)(x) * np.exp(-1/2*x**2)

    x = np.linspace(xmin, xmax, 70)

  #  print(En)

    E = []
    for n in range(0, values+1):

        E.append(n+1/2)
        psi = harm_osc(n, x)
        I = np.trapz(psi*psi, x)
       # print(I)
        plt.scatter(x, psi*psi+En[n], s=20, c='green', marker="x")

    numerical = mpatches.Patch(color='blue', label='numerical solution', linestyle='-')
    analytical = mpatches.Patch(color='green', label='analytical solution', linestyle='-')
    plt.legend(handles = [numerical, analytical], loc='upper left')
 #   plt.show()
   # plt.savefig('./plots/harmonic_oscillator.png')

    return E, En

def box(xmin, xmax, mesh, V):
    xmin=xmin
    xmax=xmax
    mesh=mesh
    V=V

    def boundary(x, on_bnd):
        return on_bnd

    bc = DirichletBC(V, Constant(0.), boundary)

    u = TrialFunction(V)
    v = TestFunction(V)

    pot_ = Expression('x[0] == xmin || x[0] == xmax ? 1000 : 0', xmin=xmin, xmax=xmax, degree=2)

    pot = interpolate(pot_, V)

    a = 1/2*inner(grad(u), grad(v))*dx + pot*u*v*dx
    m = u*v*dx

    A = PETScMatrix()
    M = PETScMatrix()

    assemble(a, tensor=A)
    assemble(m, tensor=M)

    bc.apply(A)

    # create eigensolver
    eigensolver = SLEPcEigenSolver(A, M)
    eigensolver.parameters['spectrum'] = 'smallest magnitude'

    # solve for eigenvalues
    values = 6
    eigensolver.solve(values)

    u = Function(V)

    # plot
    fig = plt.figure(figsize=[10, 10])
    En = []
    for i in range(0, values+1):
        E, E_c, R, R_c = eigensolver.get_eigenpair(i)
        En.append(E)
        x = np.linspace(xmin, xmax, len(R))

        I = np.trapz(R*R, x)
   #     print(I)

        #plot eigenfunction
        plt.plot(x, 10*R*R+E, c='blue')
        plt.plot(x, R_c+E, lw=0.5, c='black')
        plt.annotate(f'$E_{ { i } } = {round(En[i], 3)}$', (xmax-0.1, 0.05+E), fontsize=14)
        plt.annotate(f'$|\Psi _{ { i } }|²$', (xmin, 0.05+E), fontsize=14)

    plt.axvline(x=xmin, c='red')
    plt.axvline(x=xmax, c='red')
    plt.ylim(0, max(En)+0.5)
    plt.title('Particle in \'infinite\' well', fontsize=20)
    plt.xticks([])
    plt.yticks([])



    # plot analytical solutions
    x = np.linspace(xmin, xmax, 70)

 #   print(En)

    for i in range(values+1, values+2):
        E, E_c, R, R_c = eigensolver.get_eigenpair(i)
    #    En.append(E)

    E = []
    for n in range(1, values+2):

       # particle in a box
        E.append(n**2 * np.pi**2 / (2*100))
        kn = n*np.pi/10
        if n%2 == 0:
            psi = sqrt(2/10) * np.sin(kn*x)
        if n%2 == 1:
            psi = sqrt(2/10) * np.cos(kn*x)


       # plot

        I = np.trapz(psi*psi, x)
       # print(I)
        plt.scatter(x, 0.5*psi*psi+En[n-1], s=20, c='green', marker="x")



  #  plt.tight_layout()

    numerical = mpatches.Patch(color='blue', label='numerical solution', linestyle='-')
    analytical = mpatches.Patch(color='green', label='analytical solution', linestyle='-')
    plt.legend(handles = [numerical, analytical], loc='upper left')
  #  plt.show()
   # plt.savefig('./plots/box_potential.png')

    return E, En

def morse(xmin, xmax, mesh, V):
    xmin=xmin
    xmax=xmax
    mesh=mesh
    V=V
    D = 12
    a = 1

    def boundary(x, on_bnd):
        return on_bnd

    bc = DirichletBC(V, Constant(0.), boundary)

    u = TrialFunction(V)
    v = TestFunction(V)

    pot_ = Expression('D*pow((1-exp(a*(x[0]-x0))), 2)', D=D, a=a, x0=0, degree=2)
  #  pot_ = Expression('D*(exp(-2*a*(x[0]-x0))-2*exp(-a*(x[0]-x0)))', D=D, a=a, x0=0, degree=2)

    pot = interpolate(pot_, V)

    a = 1/2*inner(grad(u), grad(v))*dx + pot*u*v*dx
    m = u*v*dx

    A = PETScMatrix()
    M = PETScMatrix()

    assemble(a, tensor=A)
    assemble(m, tensor=M)

    bc.apply(A)

    # create eigensolver
    eigensolver = SLEPcEigenSolver(A, M)
    eigensolver.parameters['spectrum'] = 'smallest magnitude'

    # solve for eigenvalues
    values = 3
    eigensolver.solve(values)

    u = Function(V)

    # plot
    fig = plt.figure(figsize=[10, 10])
    En = []
    for i in range(0, values+1):
        E, E_c, R, R_c = eigensolver.get_eigenpair(i)
        En.append(E)
        x = np.linspace(xmin, xmax, len(R))

        I = np.trapz(R*R, x)
       # print(I)

        #plot eigenfunction
        plt.plot(x, 20*R*R+E, c='blue')
        plt.plot(x, R_c+E, lw=0.5, c='black')
        plt.annotate(f'$E_{ { i } } = {round(En[i], 3)}$', (xmax-0.1, 0.1+E), fontsize=14)
        plt.annotate(f'$|\Psi _{ { i } }|²$', (xmin, 0.2+E), fontsize=14)

    plt.plot(x, 12*(1-np.exp(-x))**2, c='red')
 #   plt.plot(x, 12*(np.exp(-2*x)-2*np.exp(-x)))
    plt.ylim(0, max(En)+2)
  #  plt.title('Particle in \'infinite\' well', fontsize=20)
    plt.title('Morse Potential $V(x) = D \cdot (1 - e^{- \\alpha (x - x0)})^2$', fontsize=20)
    plt.xticks([])
    plt.yticks([])





    # plot analytical solutions
    x = np.linspace(xmin, xmax, 70)

  #  print(En)

    E = []

    for n in range(0, values+1):
       # morse

        w = np.sqrt(2*12)/(2*np.pi)
        l = np.sqrt(2*D)
        z = 2*l*np.exp(-x)
        N = np.sqrt(factorial(n)*(2*l-2*n-1)/factorial(2*l-n-1))
        psi = N * z**(l-n-1/2) * np.exp(-z/2) * genlaguerre(n, 2*l-2*n-1)(z)

        E.append(-1/2*(l-n-1/2)**2 + 12)
      #  print(E)
      #  print(En)

       # plot

        I = np.trapz(psi*psi, x)
       # print(I)
        plt.scatter(x, psi*psi+En[n], s=20, c='green', marker="x")



  #  plt.tight_layout()

    numerical = mpatches.Patch(color='blue', label='numerical solution', linestyle='-')
    analytical = mpatches.Patch(color='green', label='analytical solution', linestyle='-')
    plt.legend(handles = [numerical, analytical], loc='upper left')
 #   plt.show()
#    plt.savefig('./plots/morse_potential.png')

    return E, En

def main():

    xmin = -5; xmax = 5
    s_h = []
    s_b = []
    s_m = []
    meshfinesse = []

    for n in range(20, 300, 20):
        meshfinesse.append(n)
        mesh = IntervalMesh(n, xmin, xmax)
        V = FunctionSpace(mesh, 'CG', 1)
        E_h_an, E_h_num = harmonic_oscillator(xmin, xmax, mesh, V)
        E_b_an, E_b_num = box(xmin, xmax, mesh, V)
        E_m_an, E_m_num = morse(xmin, xmax, mesh, V)

        d_h = np.subtract(E_h_an, E_h_num)/E_h_an
        s_h.append(np.average(np.abs(d_h)))

        d_b = np.subtract(E_b_an, E_b_num)/E_b_an
        s_b.append(np.average(np.abs(d_b)))

  #  print(E_m_an, E_m_num)
        d_m = np.subtract(E_m_an, E_m_num)/E_m_an
        s_m.append(np.average(np.abs(d_m)))

    s_h = np.array(s_h)
    s_b = np.array(s_b)
    s_m = np.array(s_m)
    print(s_h)
    print(s_b)
    print(s_m)

    fig = plt.figure()
    plt.plot(meshfinesse, 100*s_h, marker='x', c='green', label='harmonic oscillator')
    plt.plot(meshfinesse, 100*s_b, marker='x', c='red', label='box potential')
    plt.plot(meshfinesse, 100*s_m, marker='x', c='blue', label='morse potential')
    plt.title('Mittlerer prozentualer Fehler der Eigenwerte', fontsize=20)
    plt.ylabel('MAPE (%)')
    plt.xlabel('number of intervalls (meshsize = 10)')
#    plt.xticks(np.arange(0, 1, 0.1))
#    plt.gca().invert_xaxis()
#    plt.xscale('log')

    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
