import numpy as np
import scipy as sp
import scipy.linalg as la
import math
import time as tm


def inicialization(J0):
    """initialize couplings as J=1 for n = 3
    and based on the previous step for n > 3"""

    nj = np.shape(J0)[0]+1
    J = np.zeros(nj)

    for i in range(nj//2):
        J[i] = J0[i]
    for i in range(nj//2,nj):
        J[i] = J0[i-1]

    return J

def gradient(m,J,delta,time):
    """Loss function and gradient calculation"""

    beta = 0.5
    llambda = 1E-6
    nh = np.shape(J)[0]+1

    Jp = J.copy()
    Jp[m] += beta

    Jl = J.copy()
    Jl[m] -= beta

    fidelity_p = fidelity(Jp,nh,delta,time)
    fidelity_l = fidelity(Jl,nh,delta,time)

    loss_p = 1-fidelity_p+llambda*np.max(Jp)

    loss_l = 1-fidelity_l+llambda*np.max(Jl)

    gm = (loss_p-loss_l)/(2*beta)

    return gm

def fidelity(J, n, delta=1.0, time=False):
    """
    Returns transmission probability (|<1|n>|²) for an XXZ
    Hamiltonian in the one-excitation basis for a given set
    of couplings. It can also be used as fitness function
    for the genetic algorithm.

    Parameters:
    ----------
    - J = couplings for xxz hamiltonian
    - n = length of the chain
    - delta = anisotropy parameter (defaults to 1, i.e., Heisenberg Hamiltonian )
    - time = transmission time (if false, t = n where n is the size of the system)
    - erase_last_gene = erase last number of input vector J (to be used when
    the provided solutions have been obtained with a genetic algorithm variant
    that stores an extra gene not corresponding to a coupling)

    Returns:
    --------
    -  F = Transmission probability (|<1|n>|²) for the provided system
    """

    (eigvals, eigvects) = diag_hxxz(J, n, delta)
    if time:
        t = time
    else:
        t = n

    c1cn = np.zeros(n)

    for i in range(0, c1cn.size):
        c1cn[i] = eigvects[0, i] * eigvects[n - 1, i]

    F = 0.0
    Fr = 0.0
    Fi = 0.0

    for i in range(0, n):
        Fr = Fr + math.cos(eigvals[i] * t) * c1cn[i]
        Fi = Fi + math.sin(eigvals[i] * t) * c1cn[i]

    Fr = np.real(Fr)
    Fi = np.real(Fi)

    F = Fr * Fr + Fi * Fi

    return F

def diag_hxxz(J, n, delta=1.0):
    """
    Diagonzalizes XXZ Hamiltonian in the one-excitation basis
    from complete set of couplings (full chain is symmetric).
    Default value of the anisotropy parameter (Delta) is set to 1,
    making the default system a Heisenberg Hamiltonian.

    Parameters
    ----------
         - J: First half of couplings
         - n: length of the chain
         - delta: anisotropy parameter

    Returns
    -------
        - eigvals: n size array containing XXZ Hamiltonian
        eigenvalues
        - eigvects: nxn size array containing XXZ Hamiltonian
        eigenvectors as columns
    """

    sumj = -0.25 * np.sum(J)

    d = np.ones(n)
    ds = np.ones(n - 1)

    sumj = -0.25 * np.sum(J)

    for i in range(0, n):
        if i == 0:
            d[i] = sumj + 0.5 * J[i]
        elif i == n - 1:
            d[i] = sumj + 0.5 * J[i - 1]
        else:
            d[i] = sumj + 0.5 * J[i] + 0.5 * J[i - 1]

        d[i] = d[i] * delta

    for i in range(0, n - 1):
        ds[i] = J[i] * (-0.5)

    (eigvals, eigvects) = la.eigh_tridiagonal(d, ds)

    return eigvals, eigvects

#!--------------------------------
#! Parameter definition
#!--------------------------------

alpha = 0.05
tol = 1e-3
nmax = 40
kmax = 100000

delta = 1.


# set initial guess of J3 randomly


J0 = np.ones(3)
t0 = tm.time()
for nj in range(3,nmax):
    nh = nj+1
    time = nh
    if nj>3:
        J = inicialization(J0)
    else:
        J = J0
    t1 = tm.time()
    for k in range(kmax):
        m = np.random.randint(nj)

        gm = gradient(m,J,delta=delta,time=time)

        J[m] -= alpha*gm

        nfidelity = fidelity(J,nh,delta=delta,time=time)
        # with open(f'fidelity_evolution_{nj}.txt','a') as f:
        #     f.write(f'{k},{nfidelity}\n')
 
        if np.abs(1 - nfidelity) < tol:
            break
    t2 = tm.time()

    # with open(f'final_J_{nj}.txt','a') as f:
    #     f.write(f'{J}\n')   

    #     print(f'final J: {J}')
    J0 = J
    with open(f'final_fidelity.txt','a') as f:
        f.write(f'{nj},{nfidelity},{t2-t1} ,{t2-t0}\n')
        print(f'{nj},{nfidelity}')