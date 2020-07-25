function [Gmat,Fmat] = GFmats(rho)
%GFMATS Summary of this function goes here
%   Detailed explanation goes here

d=size(rho,1);
lambda=(GeneralizedPauliMatrices(d))./(sqrt(2));
% GFmat=sparse(d^2-1,d^2-1);
GFmat=zeros(d^2-1,d^2-1);

for i=1:(d^2-1)
    for j=1:(d^2-1)
        GFmat(i,j)=trace(lambda(:,:,i)*lambda(:,:,j)*rho);
    end
end

Gmat=real(GFmat);
Fmat=imag(GFmat);
% Gmat=full(real(GFmat));
% Fmat=sparse(imag(GFmat));

end