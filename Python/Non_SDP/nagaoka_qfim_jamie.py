import numpy as np
import qutip as qt
import itertools as it
import multiprocessing as m
import os
import time
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import minimize

"""
So this is going to be my python version for the nagaoka formulation for the quantum craemer rao bound i.e. a lower bound on the varience of estimation of unknown parameters in some given system. 

In particular I am only interested in three parameters of a magnetic field, so will make simplifications and optimisations based on this. 



"""

ide    = qt.identity(2)
sx     = qt.sigmax()
sy     = qt.sigmay()
sz     = qt.sigmaz()
si     = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)

def ham(alpha, n): # Returns the hamiltonian generator of the external single body magnetic field
    H  = (alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n))
    return H

def GHZ_3D(n): #returns the superposition of the tensor product of eigenvectors of all pauli matricies
    state = qt.tensor([d1[0]]*n) + qt.tensor([d1[1]]*n) +qt.tensor([d2[0]]*n) + qt.tensor([d2[1]]*n) +qt.tensor([d3[0]]*n) + qt.tensor([d3[1]]*n)
    return state.unit()

def SC_estim_ham(pauli, N):#sets up estimation hamiltonian for a given pauli (or particular mag field direction)
    h = qt.tensor([qt.Qobj(np.zeros((2,2)))]*N)
    for i in range(0,N):
        a    = [si]*N
        a[i] = pauli
        b    = qt.tensor(a)
        h    += b
    return h

def noise(n, gamma, rho): # takes a state and returns the state dephased (pauli-Z noise) by gamma amount
    e0    = qt.Qobj([[1, 0], [0, np.sqrt(1-gamma)]])
    e1    = qt.Qobj([[0,0],[0, np.sqrt(gamma)]])
    kraus =[x for x in it.product([e0,e1], repeat = n)]
    out   =[]
    for i in range(len(kraus)):
        out.append(qt.tensor([kraus[i][j] for j in range(n) ]))
    state = qt.tensor([qt.Qobj(np.zeros((2,2)))]*n)
    for i in range(len(out)):
        state += out[i]*rho*out[i]
    return state

def U_mag(alpha, rho, n): # this states a state and returns that state after experiancing an external magnetic field
    H      = qt.Qobj.expm(-1j*(alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n)))
    Hdag   = qt.Qobj.expm(1j*(alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n)))
    output = H*rho*Hdag
    return output

def final_state(alpha, gamma, phi, n): #returns a given state which has undergone a magnetic field and then pauli-z dephasing
    rho       = phi*phi.dag()
    rho_alpha = U_mag(alpha, rho, n)
    rho_n     = noise(n, gamma, rho_alpha)
    return rho_n

def qfim_pure(rhot, L):
    mat = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            mat[i,j] = 4*(np.real(qt.Qobj.tr(L[i]*L[j]*rhot) - qt.Qobj.tr(L[i]*rhot)*qt.Qobj.tr(L[j]*rhot)  ))
    a = np.matrix.trace(np.linalg.inv(mat))
    return a

def A(k,n):
	pauli = [qt.sigmax(),qt.sigmay(), qt.sigmaz()]
	alpha = [1e-4,2e-4,3e-4]
	L = qt.Qobj(np.zeros((2**n,2**n)), dims=([[2]*n,[2]*n]))
	for i in range(10):
		for j in range(10):
			L +=  (1j**(i+j))/(i+j+1)*(ham(alpha, n)**i)*SC_estim_ham(pauli[k], n)*((-ham(alpha, n))**j)*(1/(factorial(i)*factorial(j)))
	return L


def dlambda(rhot,alpha, gamma, phi, n,i):
    """
    so this derivative function should be changed to be closer to analytic,
    i.e. use the "A" operator, this should be more accurate and quicker, 
    however this isn't a huge problem and the second order derivatives 
    are pretty accurate


    """
    delta = 1e-8
    temp_alpha1    = alpha[:]
    temp_alpha2    = alpha[:]
    temp_alpha3    = alpha[:]
    temp_alpha4    = alpha[:]
    temp_alpha1[i] += 2*delta
    temp_alpha2[i] += delta
    temp_alpha3[i] -= delta
    temp_alpha4[i] -= 2*delta
    t1             = final_state(temp_alpha1, gamma, phi, n)
    t2             = final_state(temp_alpha2, gamma, phi, n)
    t3             = final_state(temp_alpha3, gamma, phi, n)
    t4             = final_state(temp_alpha4, gamma, phi, n)
    out              = (-t1 + (8*t2) - (8*t3) + t4)/(12*delta)

    return out


def SLD(rhot,alpha, gamma, phi, n, i):
    """
    because I am only looking at dephasing noise I should impliment the fact that the
    rank of the state is N-#0 in the state vector. This will do away with the tidy up
    tollerance. However this might still be an issue if there are very very small 
    elements of the state vector that just leads to the eigenvalues being inaccurate.


    """
    D, V = qt.Qobj.eigenstates(rhot)
    L = qt.tensor([qt.Qobj(np.zeros((2,2)))]*n)
    a = dlambda(rhot,alpha, gamma, phi, n, i)
    D = qt.Qobj(D)
    D = D.tidyup(atol=1e-7)
    for i in range(2**n):
        for j in range(2**n):
            if D[i][0][0]+D[j][0][0] == 0:
                continue
            rd = ( (V[i].dag()) * a * V[j])
            lt = ( ( (rd/(D[i][0][0]+D[j][0][0])) ) * ( V[i]*V[j].dag() ) )
            L += lt
    L=2*L
    return L

def qfim(phi,rhot,alpha, gamma, n):
    L = [0]*3
    mat = np.zeros((3,3))
    for i in range(3):
        L[i] = SLD(rhot,alpha, gamma, phi, n, i)
    for i in range(3):
        for j in range(i+1):
            mat[i,j] = mat[j,i] = np.real(qt.Qobj.tr(L[i]*L[j]*rhot))

    a = np.matrix.trace(np.linalg.inv(mat))
    #print(np.linalg.matrix_rank(mat))
    #print("qfi mat: ", mat)
    #print("     ")
    #print("qfi inv: ", np.linalg.inv(mat))
    return a



"""
So this A matrix is the 'differential operator' and is dependant on the unitary
which for my case is fixed. So I should calculate this matrix very accuractly 
and save it to a file, will save a lot of effort.

"""



def NagHolCRB(rho, D_Rho, n):

    x0     = np.random.random(3*2**n)
    cons   = {"fun": Constraint, "args": (D_Rho,), "type": "eq"}
    ops    = {"maxiter": 1000}
    sol    = minimize(NagObjFun, x0, args=(rho,), method='SLSQP', constraints = cons,options = ops )
    holCRB = sol.fun
    print(sol)
    return holCRB

def NagObjFun(x,rho):
    X = np.array_split(x, 3)
    X = [VecToObj(X[i]) for i in range(3)]
    z = np.zeros((3,3),dtype=np.complex_)
    for i in range(3): #once everythin is working contract this loop
        for j in range(3):
            z[i,j] = qt.Qobj.tr(X[i]*X[j]*rho)
    return np.matrix.trace(np.real(z)) +  np.matrix.trace(np.abs(np.imag(z)))

def BlochVec(rho):
    out = [0]*(4**n)
    for i in range(4**n):
        out[i] = float(np.real(qt.Qobj.tr(rho*comp[i])))
    return

def VecToObj(vec):
    obj = [vec[i]*comp[i] for i in range(len(vec))]
    return sum(obj)

def Constraint(x, Drhos):
    tol = 1e-7
    X = np.array_split(x, 3)
    X = [VecToObj(X[i]) for i in range(3)]
    z = np.zeros((3,3),dtype=np.complex_)
    for i in range(3): #once everythin is working contract this loop
        for j in range(3):
            temp = qt.Qobj.tr(X[i]*Drhos[j])
            #if temp - 1 < tol:
            #    temp = 1
            z[i,j] = temp
    dif = z - np.eye(3)
    out = [0]*9
    count = 0
    for i in range(3):
        for j in range(3):
            out[count] += abs(dif[i,j])
            count += 1
    #this whole thing needs to be improved, pretty sloppy code
    out = sum(out)
    if abs(out) <= tol:
        out = 0
    return out

def Drho(rhot,alpha, gamma, phi, n):
    ds = [0,0,0]
    for i in range(3):
        ds[i] = dlambda(rhot,alpha, gamma, phi, n,i)
    return ds

def bloch_comp(n):
    kraus = [x for x in it.product([si,sx,sy,sz], repeat = n)]
    comp  = []
    for i in range(len(kraus)):
        comp.append(qt.tensor([kraus[i][j] for j in range(n) ]))
    return comp

n                = 4
comp             = bloch_comp(n)
phi              = GHZ_3D(n)
alpha            = [1e-4, 2e-4, 3e-4]
gamma            = 0
rho_t            = final_state(alpha, gamma, phi, n)
Drhos            = Drho(rho_t,alpha, gamma, phi, n)





print(qfim(phi,rho_t,alpha, gamma,  n))
print(NagHolCRB(rho_t, Drhos, n))







