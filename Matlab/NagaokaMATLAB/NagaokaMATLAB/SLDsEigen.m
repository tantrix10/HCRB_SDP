function [SLDs] = SLDsEigen(rho,drhovec,tol)
%SLDSEIGEN Summary of this function goes here
%   Detailed explanation goes here

if nargin<3
    tol=1.0e-8; %tolerance in considering the eigenvalues of the state zero
end

d=size(rho,1);
npar=size(drhovec,3);

[V,D]=eig(rho);
eigrho=diag(D);

SLDs=zeros(d,d,npar);

for q=1:npar
    for i=1:d
        for j=1:d
            if (eigrho(i)+eigrho(j)) > tol
                SLDs(:,:,q) = 2.*(V(:,i)'*drhovec(:,:,q)*V(:,j)).*(V(:,i)*V(:,j)')./(eigrho(i)+eigrho(j)) + SLDs(:,:,q);           
            end
        end
    end
end