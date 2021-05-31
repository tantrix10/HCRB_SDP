import numpy as np
import qutip as qt
from scipy.optimize import minimize
import itertools as it
import multiprocessing as m
import os
import time
import matplotlib.pyplot as plt
#from progressbar import ProgressBar
#pbar = ProgressBar()

ide    = qt.identity(2)
sx     = qt.sigmax()
sy     = qt.sigmay()
sz     = qt.sigmaz()
si     = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)
zo = up= qt.Qobj([[1],[0]])
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

def ghz(n):
    state = qt.tensor([d1[0]]*n) + qt.tensor([d1[1]]*n) +qt.tensor([d2[0]]*n) + qt.tensor([d2[1]]*n) +qt.tensor([d3[0]]*n) + qt.tensor([d3[1]]*n)
    return state.unit()

def SC_estim_ham(pauli, N):#sets up estimation hamiltonian for a pauli
    h = qt.tensor([qt.Qobj(np.zeros((2,2)))]*N)
    for i in range(0,N):
        a    = [si]*N
        a[i] = pauli
        b    = qt.tensor(a)
        h    += b
    return h

def noise(n, gamma, rho):
    e0    = qt.Qobj([[1, 0], [0, np.sqrt(1-gamma)]])
    e1    = qt.Qobj([[0,0],[0, np.sqrt(gamma)]])
    #e0 = np.sqrt((1-gamma/2))*si
    #e1 = np.sqrt((gamma/2))*sz
    kraus =[x for x in it.product([e0,e1], repeat = n)]
    out   =[]
    for i in range(len(kraus)):
        out.append(qt.tensor([kraus[i][j] for j in range(n) ]))
    state = qt.tensor([qt.Qobj(np.zeros((2,2)))]*n)
    for i in range(len(out)):
        state += out[i]*rho*out[i]

    return state



def hamiltonian(alpha, rho, n): # I must change this in order to allow alpha to be of different sizes
    H      = qt.Qobj.expm(-1j*(alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n)))
    Hdag   = qt.Qobj.expm(1j*(alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n)))
    output = H*rho*Hdag
    return output


def final_state(alpha, gamma, phi, n):
    rho       = phi*phi.dag()
    rho_alpha = hamiltonian(alpha, rho, n)
    rho_n     = noise(n, gamma, rho_alpha)
    return rho_n


def dlambda(rhot,alpha, gamma, phi, n,i):
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


def qfi(rhot,alpha, gamma, phi, n, i):
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
        L[i] = qfi(rhot,alpha, gamma, phi, n, i)
    for i in range(3):
        for j in range(3):
            mat[i,j] = np.real(qt.Qobj.tr(L[i]*L[j]*rhot))
    a = np.matrix.trace(np.linalg.inv(mat))
    #print(np.linalg.matrix_rank(mat))
    #print("qfi mat: ", mat)
    #print("     ")
    #print("qfi inv: ", np.linalg.inv(mat))
    return a

def weak_com(rhot,alpha, gamma, phi, n):
    L = [0]*3
    for i in range(3):
        L[i] = qfi(rhot,alpha, gamma, phi, n, i)
    results = []
    a = qt.Qobj.tr(rhot*(L[0]*L[1]-L[1]*L[0]) )
    b = qt.Qobj.tr(rhot*(L[0]*L[2]-L[2]*L[0]) )
    c = qt.Qobj.tr(rhot*(L[1]*L[0]-L[0]*L[1]) )
    #return a,b,c
    return [np.imag(a),np.imag(b),np.imag(c)]

"""
n                = 2
phi              = ghz(n)


alpha            = [1e-4,10,10]
gamma            = 0
rhot            = final_state(alpha, gamma, phi, n)

print(weak_com(rhot,alpha, gamma, phi, n)[0])
"""
n               = 2
phi             = ghz(n)
alpha            = [[1e-4, 1e-4, 1e-4],[1e-4, 1e-4, 1],[1, 1e-4, 1],[1, 1, 1]]
gamma            = np.linspace(0,0.9,10)

a201 =[]
a401 = []
a501 = []

a211 =[]
a411 = []
a511 = []

a221 =[]
a421 = []
a521 = []

a231 =[]
a431 = []
a531 = []

a202 =[]
a402 = []
a502 = []

a212 =[]
a412 = []
a512 = []

a222 =[]
a422 = []
a522 = []

a232 =[]
a432 = []
a532 = []


a203 =[]
a403 = []
a503 = []

a213 =[]
a413 = []
a513 = []

a223 =[]
a423 = []
a523 = []

a233 =[]
a433 = []
a533 = []






ns = [2,4,5]

for k in range(3):
    for j in range(4):
        for i in range(len(gamma)):
            phi             = ghz(ns[k])
            rhot            = final_state(alpha[j], gamma[i], phi, ns[k])
            out,out2,out3 = weak_com(rhot,alpha[j], gamma[i], phi, ns[k])
            print(out,out2,out3)
            eval('a'+str(ns[k])+str(j)+'1').append(out)
            eval('a'+str(ns[k])+str(j)+'2').append(out2)
            eval('a'+str(ns[k])+str(j)+'3').append(out3)
            #a2.append(out2)
            #a3.append(out3)


fix, ax = plt.subplots(4,3)


ax[0,0].plot(gamma, a201)
ax[0,1].plot(gamma, a401)
ax[0,2].plot(gamma, a501)

ax[1,0].plot(gamma, a211)
ax[1,1].plot(gamma, a411)
ax[1,2].plot(gamma, a511)

ax[2,0].plot(gamma, a221)
ax[2,1].plot(gamma, a421)
ax[2,2].plot(gamma, a521)

ax[3,0].plot(gamma, a231)
ax[3,1].plot(gamma, a431)
ax[3,2].plot(gamma, a531)


ax[0,0].plot(gamma, a202)
ax[0,1].plot(gamma, a402)
ax[0,2].plot(gamma, a502)

ax[1,0].plot(gamma, a212)
ax[1,1].plot(gamma, a412)
ax[1,2].plot(gamma, a512)

ax[2,0].plot(gamma, a222)
ax[2,1].plot(gamma, a422)
ax[2,2].plot(gamma, a522)

ax[3,0].plot(gamma, a232)
ax[3,1].plot(gamma, a432)
ax[3,2].plot(gamma, a532)

ax[0,0].plot(gamma, a203)
ax[0,1].plot(gamma, a403)
ax[0,2].plot(gamma, a503)

ax[1,0].plot(gamma, a213)
ax[1,1].plot(gamma, a413)
ax[1,2].plot(gamma, a513)

ax[2,0].plot(gamma, a223)
ax[2,1].plot(gamma, a423)
ax[2,2].plot(gamma, a523)

ax[3,0].plot(gamma, a233)
ax[3,1].plot(gamma, a433)
ax[3,2].plot(gamma, a533)



ax[0,0].set_title('2 qubits')
ax[0,1].set_title('4 qubits')
ax[0,2].set_title('5 qubits')

ax[0,0].set_ylabel('alpha = [0,0,0]')
ax[1,0].set_ylabel('alpha = [0,0,1]')
ax[2,0].set_ylabel('alpha = [1,0,1]')
ax[3,0].set_ylabel('alpha = [1,1,1]')

ax[3,1].set_xlabel('noise')


#plt.xlabel('noise')
#plt.ylabel('weak com')
#plt.title(str(n)+' qubits'+str(alpha)+' para')
plt.show()
