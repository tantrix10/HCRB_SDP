import numpy as np
import qutip as qt
import itertools as it
import multiprocessing as m
import os
import time
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

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
zo = qt.Qobj([[1],[0]])
on = qt.Qobj([[0],[1]])


def new(n,a):
    combin = [on]*n
    state = a*(qt.tensor([zo]*n))
    for i in range(n):
        temp = combin[:]
        temp[i] = zo
        temp = qt.tensor(temp)
        state += temp
    return state.unit()


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


def dlambda(rhot,alpha, gamma, phi, n, i):
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
    #print("before", D)
    if gamma > 0:
        rank = 0
        for i in range(2**n):
            if abs(phi[i][0][0]) != 0:
                rank +=1
        D[:(2**n - rank)] = 0
        D = qt.Qobj(D)
    elif gamma == 0:
        D = qt.Qobj(D)
        D = D.tidyup(atol=1e-3)
    #print("after",D)
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
    ds     = np.vstack((BlochVec(D_Rho[0]),BlochVec(D_Rho[1]),BlochVec(D_Rho[2])    ))
    x0     = np.random.random((3*2**(2*n)))#.reshape(3*2**(2*n),1)

    A      = np.kron(np.eye(3),ds)
    B      = np.eye(3).reshape(9)

    cons   = LinearConstraint(A,B,B)
    ops    = {"maxiter": 1000}
    sol    = minimize(NagObjFun, x0, args=(rho,), method='SLSQP', constraints = [cons],options = ops)

    holCRB = sol.fun
    #print(sol.status)
    return holCRB

def NagObjFun(x,rho):
    X = np.array_split(x, 3)
    X = [VecToObj(X[i]) for i in range(3)]
    z = np.zeros((3,3),dtype=np.complex_)
    for i in range(3): #once everythin is working contract this loop
        for j in range(3):
            z[i,j] = qt.Qobj.tr(X[i]*X[j]*rho)
    zim = np.imag(z)
    zieg = 2*np.sqrt(np.real(zim[0,1])**2 + np.real(zim[0,2])**2 + np.real(zim[2,1])**2 )
    #zieg = sum(np.abs(np.linalg.eigvals(z)))
    out = np.matrix.trace(np.real(z)) +  zieg
    return out

def BlochVec(rho):
    out = [0]*(4**n)
    for i in range(4**n):
        out[i] = float(np.real(qt.Qobj.tr(rho*comp[i])))
    return out

def VecToObj(vec):
    obj = [vec[i]*comp[i] for i in range(len(vec))]
    return sum(obj)

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





n                = 2
comp             = bloch_comp(n)



phi              = GHZ_3D(n)

#phi = [[-0.30639856-0.18134644j], [ 0.79257257-0.24262545j], [ 0.27494254-0.13230307j], [ 0.27494254-0.13230307j]]
#phi = qt.Qobj(phi, dims= ([2,2],[1,1]))

#phi              = new(n, np.sqrt(n-2))

#phi = [[-0.57801381-0.39088574j], [-0.00593478-0.06049978j], [-0.00622131-0.06342071j], [-0.00622131-0.06342071j], [-0.00622131-0.06342071j], [-0.00786648-0.08019173j], [-0.00594354-0.0605891j ], [-0.3478171 +0.60502443j]]

#phi = [[0.        +0.j        ], [0.45911933+0.1868109j ], [0.45911933+0.1868109j ], [0.        +0.j        ], [0.46167941+0.18785257j], [0.        +0.j        ], [0.        +0.j        ], [0.47247042+0.19224332j]]

#phi   = qt.Qobj([[0.4134113 +0.02897776j], [0.47565554+0.09958154j], [0.47565554+0.09958154j], [0.02086672+0.59622699j]],dims=([2,2],[1,1]))


#phi = qt.Qobj([[ 0.54582893-0.44971449j], [-0.51751788+0.24856172j], [ 0.39097275-0.13175842j], [ 0.        +0.j        ]], dims = ([[2,2],[1,1]]))


#phi = qt.Qobj(phi, dims = ([2]*3,[1]*3)).unit()
print(phi)



alpha            = [1e-4]*3
gamma            = 0
rho_t            = final_state(alpha, gamma, phi, n)
Drhos            = Drho(rho_t,alpha, gamma, phi, n)


"""
qt.fileio.qsave(rho_t, "final_state")
qt.fileio.file_data_store("rhot", rho_t)
qt.fileio.file_data_store("Dx", Drhos[0])
qt.fileio.file_data_store("Dy", Drhos[1])
qt.fileio.file_data_store("Dz", Drhos[2])

"""
print("SLD QFIM: ",qfim(phi,rho_t,alpha, gamma,  n))

a = time.time()
print("NAG QFIM: ",NagHolCRB(rho_t, Drhos, n))
b = time.time()
print(b-a)


