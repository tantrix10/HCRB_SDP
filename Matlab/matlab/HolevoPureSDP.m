function [holCRB,status] = HolevoPureSDP(psi,dpsi,W)
d=size(psi,1);
npar=size(dpsi,2);

psidpsi = psi' * dpsi;    %1 x n matrix
pardpsi = psi  * psidpsi; %dxn matrix, columns are the parallel component of the derivatives
Lmat    = 2 * ( dpsi - pardpsi );
cvx_begin sdp
    variable V(npar,npar) symmetric
    variable X(d,npar) complex
    minimize trace(sqrtW*V*sqrtW)
    subject to
        [ V , X' ; X , eye(d)] >= 0
        real(Lmat'*X)==eye(npar)
        psi'*X==zeros(1,npar)
cvx_end

holCRB=cvx_optval;
status=cvx_status;

end
