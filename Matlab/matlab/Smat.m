function [S] = Smat(rho)
%GFMATS Summary of this function goes here
%   Detailed explanation goes here

d=size(rho,1);
lambda=(GeneralizedPauliMatrices(d))./(sqrt(2));
% GFmat=sparse(d^2-1,d^2-1);
S=zeros(d^2-1,d^2-1);

for i=1:(d^2-1)
    for j=1:(d^2-1)
        S(i,j)=trace(lambda(:,:,i)*lambda(:,:,j)*rho);
    end
end
end