import qutip     as qt
import numpy     as np
import cvxpy     as cp
import itertools as it
import scipy     as sci
import numpy.matlib
import os


def rank(D, tol = 1e-9):
    """
    args:
    -D: numpy array, Eigenvalues of rho

    defualt args:
    -tol: float, tollerence of what constitues a "zero" Eigenvalue

    returns:
    -Dnonzero, numpy array, of non-zero Eigenvalues
    -Rank, int, rank of rho
    """

    mask = D > tol
    Dnonzero = D[mask]
    rank = sum(mask)

    return Dnonzero, rank



def SmatRank(snonzero, d, rnk, dim):
    #print(snonzero,d)
    mask    = np.triu(np.ones([rnk], dtype= bool),1)
    scols = np.zeros((rnk,rnk))
    for i in range(rnk):
        for j in range(rnk):
            scols[i,j] = np.real(snonzero[j])

    srows   = scols.transpose()
    siminsj = -srows + scols
    siplsj  = scols + srows

    diagS   = np.concatenate((snonzero.transpose(),siplsj[mask].transpose(),siplsj[mask].transpose(),np.matlib.repmat(snonzero.transpose(),2*(d-rnk),1).flatten() ))

    Smat = sci.sparse.spdiags(diagS,0,dim,dim)
    Smat = sci.sparse.csr_matrix(Smat)
    Smat = Smat.todense()
    Smat = np.matrix(Smat, dtype = complex)

    if rnk != 1:
        offdRank = 1j* sci.sparse.spdiags(siminsj[mask],0,int((rnk**2-rnk)/2),int((rnk**2-rnk)/2)).todense()
    else:
        offdRank = 0

    offdKer = -1j*sci.sparse.spdiags(np.matlib.repmat(snonzero,d-rnk,1).flatten(),0,rnk*(d-rnk),rnk*(d-rnk)).todense()   


    Smat[int(rnk+(rnk**2-rnk)/2):int(rnk+(rnk**2-rnk))                        ,int(rnk):int(rnk+(rnk**2-rnk)/2)]                                      =-offdRank
    Smat[int(rnk):int(rnk+(rnk**2-rnk)/2)                                     ,int(rnk+(rnk**2-rnk)/2):int(rnk+(rnk**2-rnk))]                         = offdRank
    Smat[int(rnk+(rnk**2-rnk)+rnk*(d-rnk)):int(rnk+(rnk**2-rnk)+2*rnk*(d-rnk)),int(rnk+(rnk**2-rnk)):int(rnk+(rnk**2-rnk)+rnk*(d-rnk))]               =-offdKer
    Smat[int(rnk+(rnk**2-rnk)):int(rnk+(rnk**2-rnk)+rnk*(d-rnk))              ,int(rnk+(rnk**2-rnk)+rnk*(d-rnk)):int(rnk+(rnk**2-rnk)+2*rnk*(d-rnk))] = offdKer

    return Smat



def Rmat(S, tol = 1e-8):
    
    d,v    = np.linalg.eig(S)
    
    ind    = d.argsort()[::-1]
    v      = v[:,ind]
    d      = d[ind]
    rank   = sum([d[i]>tol for i in range(len(d))])
    dclean = d[:rank]
    vclean = v[:,:rank]
    dout   = np.diag(dclean)

    return np.sqrt(dout)@ vclean.transpose().conjugate()




def naghol_spd(rho, drho, d):
    rho = (rho.dag()+rho)/2
    npar = 3
 
    D, Vi = np.linalg.eigh(rho.full())

    D = np.real(D)

    Vi = Vi[:,::-1]
    D = D[::-1]
    
    Vi = qt.Qobj(Vi,dims= [[2]*n,[2]*n])

    snonzero, rnk = rank(D)

    maskDiag = np.diag(np.ndarray.flatten(np.concatenate((np.ones([rnk,1],dtype = bool),np.zeros([d-rnk,1],dtype = bool)))))
    maskRank = np.concatenate((np.concatenate((np.triu(np.ones(rnk,dtype = bool),1), np.zeros([rnk,d-rnk],dtype = bool)),axis = 1),np.zeros([d-rnk,d],dtype = bool)))
    maskKern = np.concatenate((np.concatenate((np.zeros([rnk,rnk],dtype = bool),np.ones([rnk,d-rnk],dtype = bool)),axis = 1),np.zeros([d-rnk,d],dtype = bool)))

    fulldim = 2*rnk*d-rnk**2

    drhomat = np.zeros((fulldim,npar),dtype=np.complex_)

    for i in range(npar):
        drho[i] = (drho[i].dag()+drho[i])/2
        eigdrho = (Vi.dag())*drho[i]*Vi
        eigdrho = eigdrho.full()
        ak = eigdrho[maskKern]
        ak = ak.reshape((rnk,d-rnk)).transpose()
        ak = ak.reshape((rnk*(d-rnk)))

 
        row = np.concatenate((eigdrho[maskDiag],np.real(eigdrho[maskRank]),np.imag(eigdrho[maskRank]),np.real(ak),np.imag(ak)))
        drhomat[:,i] =  row

    S   = SmatRank(snonzero,d, rnk, fulldim)
    S   = (S.transpose().conjugate()+S)/2

    R = Rmat(S)
    
    effdim = R.shape[0]
    idd = np.diag(np.ndarray.flatten(np.concatenate((np.ones((rnk)),2*np.ones((fulldim-rnk))))))

    V = cp.Variable((npar,npar),PSD =True)
    X = cp.Variable((fulldim,npar))

    A = cp.vstack([ cp.hstack([V , X.T @ R.conjugate().transpose()]) , cp.hstack([R @ X , np.identity(effdim) ])])

    constraints = [ cp.vstack([   cp.hstack([cp.real(A),-cp.imag(A)]), cp.hstack([cp.imag(A),cp.real(A)])   ]) >> 0,  
                                 X.T @ idd @ drhomat == np.identity(3)]

    obj = cp.Minimize(cp.trace(V))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.SCS)#, verbose = True)
    out = prob.value
    return out



