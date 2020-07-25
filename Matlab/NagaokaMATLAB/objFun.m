function [objVal] = objFun(x,gell,rho,drhovec,W,d,npar)
%OBJFUN Summary of this function goes here
%   Detailed explanation goes here

H=zeros(d,d);
for i=1:(d^2-1)
    H = H + x(i).*gell(:,:,i);
end
H=H+x(d^2)*eye(d);

U=expm(-1j*H);

pvec=real(diag(U'*rho*U));
dpvec=zeros(d,npar);
for i=1:npar
    dpvec(:,i)=real(diag(U'*drhovec(:,:,i)*U));
end

CFIM=(diag(1./pvec)*dpvec)'*dpvec;

objVal=scalarCRB(CFIM,W);

end

