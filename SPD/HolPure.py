import qutip     as qt
import numpy     as np
import cvxpy     as cp


def naghol_spd(phi, dphi, d, solve = 'MOSEK', verbose_state = False):
    """
    args:
    -phi : qutip quantum object, pure normalised state, of size (d, 1)
    -dphi: qutip quantum object, vector of derivatives, of size (d, 1) x npar, where npar is the number of parameters
    -d   : integer             , dimension of Hilbert space
    
    default args:
    -solver        : string, string of solver, default is MOSEK, can also be CVXOPT, SCS. More details: https://www.cvxpy.org/tutorial/advanced/index.html 
    -verbose_state : bool  , option if you want the cvxpy solver to turn maximal information or not. verbose_state = true prints max info to terminal       

    returns: float, Holevo-Cramer-Rao bound of the statistical model defined by the encoded state and derivatives

    """
    #dictionary of SDP solvers
    solver_options = {'MOSEK':cp.MOSEK, 'CVXOPT': cp.CVXOPT, 'SCS':cp.SCS }

    #define the number of parameters
    npar = len(dphi)

    #here we define the matrix of horizontal lifts-Lmat
    #The horizontal lift is defined as: 2(|\partial_i \phi>  − <\phi| \partial_i \phi>|\phi> ) 
    psidpsi = phi.dag()*dphi
    pardphi = phi * psidpsi

    Lmat    = 2 * (dphi-pardphi)

    #convert quantum objects to numpy arrays
    #n.b. this should be sped up by just calling phi.full(), will do this and test asap
    psi = np.zeros(d,dtype= complex)
    for i in range(d):
        psi[i] = phi[i][0][0]

    #same here as above with horizontal lifts
    Lmatt = np.zeros((d,npar),dtype = complex)
    for i in range(d):
        for j in range(npar):
            Lmatt[i,j] = Lmat[j][i] 

    #set up SDP as defined in the readme/paper

    #set up variables 'variance matrix' V and 'derivative variable matrices' X
    V = cp.Variable((npar,npar),PSD =True)
    X = cp.Variable((d,npar),complex = True)

    #set up SD-matrix to optimise
    A = cp.vstack([ cp.hstack([V , X.H ]) , cp.hstack([X , np.identity(d) ])  ])


    #constraints need to be split because cvxpy will only accept a real valued SD-matrix
    constraints = [ cp.vstack([   cp.hstack([cp.real(A),-cp.imag(A)]), cp.hstack([cp.imag(A),cp.real(A)])   ]) >> 0, 
                    cp.real(X.H @ Lmatt) == np.identity(3),
                     ((psi.transpose()).conjugate() @ X) == 0]

    #define objective function
    obj = cp.Minimize(cp.trace(V))

    #define problem
    prob = cp.Problem(obj,constraints)

    #solve problem
    prob.solve(solver = solver_options[solve], verbose = verbose_state)
    
    #extract and return value
    out = prob.value

    return out


