function [holCRB] = func(X0,gamma,n)
%PURENAGAOKAHOLEVOCRB Summary of this function goes here
%   Detailed explanation goes here
if size(X0,1) == 1
    X0  = X0';
end
m   = size(X0,1);
phi = X0(1:m/2)+1j*X0(m/2 +1:m);
phi = phi/sqrt(phi'*phi);
tol=1.0e-9; %tolerance in considering the eigenvalues of the state zero
%n = 3;

alpha   = [1e-4,1e-4,1e-4];
%gamma   = 0;
rho     = final_state(alpha, gamma, phi, n);
rho     = (rho'+rho)/2;
rho     = rho/trace(rho);
drhovec = deriv(alpha, gamma, phi, n);
d       = size(rho,1);
npar    = size(drhovec,3); %drhovec is assumed to be an array of size (d,d,npar)
W       = eye(npar);


%diagonalization of the state
[V,D]      = eig(rho);
[svec,ind] = sort(diag(D),'descend');
V          = V(:,ind);

snonzero   = svec(svec>tol); %vector containing non-zero eigenvalues
rnk        = length(snonzero);    %rank of the density matrix

maskDiag = diag([true(rnk,1);false(d-rnk,1)]);
maskRank = [triu(true(rnk),1),false(rnk,d-rnk);false(d-rnk,d)];
maskKern = [false(rnk),true(rnk,d-rnk);false(d-rnk,d)];

fulldim=2*rnk*d - rnk^2;

drhomat=zeros(fulldim,npar);
for n=1:npar
   eigdrho=V'*drhovec(:,:,n)*V;
    drhomat(:,n)=[eigdrho(maskDiag);real(eigdrho(maskRank));imag(eigdrho(maskRank));real(eigdrho(maskKern));imag(eigdrho(maskKern))];
end

S=full(SmatRank(snonzero,d));
R=S^(0.5);
%R=cholcov(S);

sqrtW=W^(0.5); %#ok<NASGU>

id=diag([ones(rnk,1);2*ones(fulldim-rnk,1)]);

cvx_begin sdp quiet
    variable V(npar,npar) symmetric
    variable X(fulldim,npar)
    minimize trace(sqrtW*V*sqrtW)
    subject to
        [ V , X'*R' ; R*X , eye(fulldim)] >= 0;
        drhomat'*id*X==eye(npar);
cvx_end

holCRB=cvx_optval;
%status=cvx_status;

end
