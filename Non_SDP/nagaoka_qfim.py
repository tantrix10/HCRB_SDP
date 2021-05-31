import numpy as np
import qutip as qt
from scipy.optimize import minimize
import itertools as it
import multiprocessing as m
import os
import time
import matplotlib.pyplot as plt
from scipy.special import factorial

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
        for j in range(3):
            mat[i,j] = np.real(qt.Qobj.tr(L[i]*L[j]*rhot))
"""

this loop can be contracted because the QFIM is symmetric, I should make sure that
everything works before messing with stuff though.

"""
    a = np.matrix.trace(np.linalg.inv(mat))
    #print(np.linalg.matrix_rank(mat))
    print("qfi mat: ", mat)
    print("     ")
    print("qfi inv: ", np.linalg.inv(mat))
    return a







"""
So this A matrix is the 'differential operator' and is dependant on the unitary
which for my case is fixed. So I should calculate this matrix very accuractly 
and save it to a file, will save a lot of effort.

"""


"""

As a general thing in this code I need to do a general check on matrix mulitplcation to make sure 
that it is not element wise  as well as data types e.g. qubit qobj and numpy array

"""


def NagHolCRB(rho, D_Rho_Vec, W):
    tol = 1e-8
    d       = rho.dims[0][0]
    N_para  = 3
    W       = qt.identity(N_para)
    s       = BlochVec(rho)
    dsvec   = zeros(d^2-1,npar)
    sd      = [BlochVec(D_Rho)[i] for i in range(3)]
    for i in range(3):  
        for j in range(d**2 -1):
            dsvec[j,i] = sd[j,i]
    Gmat, Fmat = Gmats(d,rho)
    Qmat       = Gmat - np.kron(s, s.transpose())
    lvec       = np.zeros((d**2, npar))
    if np.det(Gmat) > tol:
        lvec(2:,:) = Qmat*dsvec
        lvec(1,:)  = -s.transpose()*lvec(2:,:) 
    else:
        for i in range(npar):
            Lops       = SLDsEigen(rho,drhovec,tol)
            lvec(2:,i) = StateToBloch(Lops(:,:,i)); 
            lvec(1,i)  = np.trace((np.identity(d)/d)*Lops(:,:,i));
    alf = AlphaFull(Gmat,s)
    v,d = np.linalg.eig(alf) # check that these are orthonormal
    MMat = np.real(np.linalg.solve(Qbasis, lvec)
    
    Fmatfull       =np.hstack((np.vstack((np.zeros(1,d^2), np.zeros(d^2-1,1))) , Fmat))
    FmatfullONbasis = np.real(Qbasis.transpose()*Fmatfull*Qbasis);

    Acon=np.kron(np.identity(npar),MMat.transpose())
    bcon=np.identity(npar);
    bcon=bcon(:); # need to sort out this function

    fNag = @(x)real(NagaokaObjFun(x,W,FmatfullONbasis,d**2));
    x0=rand(dim*npar,1);n');
    options = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunctionEvaluations',100000);
    Xopt,holCRB = fmincon(fNag,x0,[],[],Acon,bcon,[],[],[],options);

    Xopt=reshape(Xopt,[dim,npar]);
    return Xopt, holCRB

def NagObjFun(X,W,F,d):
    npar = 3
    X    = np.reshape(x,[d,3])
    ReV  = X.transpose()*X
    ImV  = X.tranpose()*F*X 
    return np.trace(W*ReV) +np.sum(np.abs(np.linalg.eigs(W*ImV)))

def BlochVec(rho):
    out = [0]*(4**n - 1)
    for i in range(4**n -1):
        out[i] = float(np.real(qt.Qobj.tr(rho*comp[i])))
    return

def GFmats(d, rho):
    dim      = d**2 - 1
    GFout    = np.zeros(dim, dim)
    CompNorm = [comp[i]/np.sqrt(2) for i in range(len(comp))]
    for i in range(dim):
        for j in range(dim):
            GFout[i,j] = qt.Qobj.tr(CompNorm[i]*CompNorm[j]*rho)
    Gmat = np.real(GFout)
    Fmat = np.imag(GFout)
    return Gmat, Fmat

def DeltaFull(f):
    return scipy.linalg.block_diag(0,f)

def alphaFull(g,s):
    alpha = scipy.linalg.block_diag(1,g)
    for i in range(alpha.shape[-1]-1):
        alpha[0,i+1] = s[i]
        alpha[i+1,0] = s[i]
    return

def SLDeig(rhot,alpha, gamma, phi, n, i):
    #calling these two functions like this is unessarcery but it's part of a process for now...
    return SLD(rhot,alpha, gamma, phi, n, i)

kraus = [x for x in it.product([si,sx,sy,sz], repeat = n)]
comp  = []
for i in range(len(kraus)):
    comp.append(qt.tensor([kraus[i][j] for j in range(n) ]))




n                = 2
phi              = til_state(n)
alpha            = [0.001, 0.001, 0.001]
gamma            = 0.1
rho_t            = final_state(alpha, gamma, phi, n)



print(qfim(phi,rho_t,alpha, gamma,  n))
print(NagHolCRB())




#test





