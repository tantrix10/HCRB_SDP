function [holCRB,status] = HolevoCRB_NagSDP(rho,drhovec,W,tol)
%PURENAGAOKAHOLEVOCRB Summary of this function goes here
%   Detailed explanation goes here

if nargin<4
    tol=1.0e-9; %tolerance in considering the eigenvalues of the state zero
end

d=size(rho,1);
npar=size(drhovec,3); %drhovec is assumed to be an array of size (d,d,npar)
if nargin==2
    W=eye(npar);
end

%diagonalization of the state
[V,D]=eig(rho);
[svec,ind] = sort(diag(D),'descend');
V=V(:,ind);

snonzero=svec(svec>tol); %vector containing non-zero eigenvalues
rnk=length(snonzero);    %rank of the density matrix

% mask=triu(true(rnk),1);  %select only the elements in the upper triangular part of the rank (without diagonal)
maskDiag=diag([true(rnk,1);false(d-rnk,1)]);
maskRank=[triu(true(rnk),1),false(rnk,d-rnk);false(d-rnk,d)];
maskKern=[false(rnk),true(rnk,d-rnk);false(d-rnk,d)];

fulldim=2*rnk*d - rnk^2;

drhomat=zeros(fulldim,npar);
for n=1:npar
   eigdrho=V'*drhovec(:,:,n)*V
%    eigdrhoRnk=eigdrho(1:rnk,1:rnk);
%    eigdrhoOD=eigdrho(1:rnk,rnk+1:rnk+(d-rnk))
%    drhomat(:,n)=[diag(eigdrhoRnk);real(eigdrhoRnk(mask));imag(eigdrhoRnk(mask));real(eigdrhoOD(:));imag(eigdrhoOD(:))];
    drhomat(:,n)=[eigdrho(maskDiag);real(eigdrho(maskRank));imag(eigdrho(maskRank));real(eigdrho(maskKern));imag(eigdrho(maskKern))];
end

S=SmatRank(snonzero,d);
R=cholcov(S);
effdim=size(R,1);
% R=R.*(abs(R)>tol); %this chops the matrix elemens below threshold

%S=full(SmatRank(snonzero,d));
%R=sqrtm(S);
%R=S^(0.5);
%R=[R; zeros(fulldim-rank(S),fulldim)]; 

% sqrtW=W^(0.5);
sqrtW=sqrtm(W);

id=diag([ones(rnk,1);2*ones(fulldim-rnk,1)]);

% cvx_solver mosek
%cvx_solver sdpt3
% cvx_solver sedumi
% cvx_precision('low')

cvx_begin sdp
    variable V(npar,npar) symmetric
    variable X(fulldim,npar)
    minimize trace(sqrtW*V*sqrtW)
    subject to
%         [ V , X'*R' ; R*X , eye(fulldim)] >= 0
        [ V , X'*R' ; R*X , eye(effdim)] >= 0
        drhomat'*id*X==eye(npar)
%         snonzero'*X(1:rnk,:)==zeros(1,npar)
%         [snonzero; zeros(fulldim-rnk,1)]'*X==zeros(1,npar)
cvx_end

holCRB=cvx_optval;
status=cvx_status;

end