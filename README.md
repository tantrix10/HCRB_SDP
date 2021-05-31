# HCRB_SDP
Semidefinite program (SDP) implementation of the Holevo-Cramer-Rao bound. 

Original Matlab implementation by @falbarelli, this is my python port

[Paper](https://arxiv.org/abs/1906.05724) with more details 

HolPure uses simplifications available to pure state for calculation of the HCRB, where as HolSDP is for general states.


Both HolPure and HolSDP implement a 'naghol_sdp' function that takes
- phi/rho: (for pure/general respectively), the final parameter encoded state
- dphi: a vector of derivatives of the parameter encoded state wrt to each parameter
- d: dimension of Hilbert space

optional args
- solve (MOSEK, CVXOPT, SCS): selects the SDP solver to use (we tend to use MOSEK for pure state and SCS for general)
- verbose_state (bool): turn verbose messaging from the solver on/off

# TODO

1. Tidy up matlab implementation (mainly just removing the libraries I accidentally dumped into the repo, in a branch)
2. Tighten up tests and add back into repo
3. Code review (I know I need to make the imports explicit, comments for example)
4. Improve this readme!
5. Add examples of use in