import qutip     as qt
import numpy     as np
import cvxpy     as cp
import itertools as it
import scipy     as sci
import numpy.matlib
import os
from tabulate import tabulate


ide    = qt.identity(2)
sx     = qt.sigmax()
sy     = qt.sigmay()
sz     = qt.sigmaz()
si     = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)
zo     = qt.Qobj([[1],[0]])
on     = qt.Qobj([[0],[1]])





def new(n,a):
    combin = [on]*n
    state = a*(qt.tensor([zo]*n))
    for i in range(n):
        temp = combin[:]
        temp[i] = zo
        temp = qt.tensor(temp)
        state += temp
    return state.unit()


def GHZ_3D(n): #returns the superposition of the tensor product of eigenvectors of all pauli matricies
    state = qt.tensor([d1[0]]*n) + qt.tensor([d1[1]]*n) +qt.tensor([d2[0]]*n) + qt.tensor([d2[1]]*n) +qt.tensor([d3[0]]*n) + qt.tensor([d3[1]]*n)
    return state.unit()


def ham(alpha, n): # Returns the hamiltonian generator of the external single body magnetic field
    H  = (alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n))
    return H


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
    U      = qt.Qobj.expm(-1j*(alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n)))
    #Udag   = qt.Qobj.expm(1j*(alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n)))
    output = U*rho*U.dag()
    return output

def final_state(alpha, gamma, phi, n): #returns a given state which has undergone a magnetic field and then pauli-z dephasing
    
    rho       = phi*phi.dag()
    rho = (rho.dag()+rho)/2
    rho = rho/qt.Qobj.tr(rho)
    rho_alpha = U_mag(alpha, rho, n)
    rho_n     = noise(n, gamma, rho_alpha)
    return rho_n


def dlambda(rhot,alpha, gamma, phi, n, i):
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

