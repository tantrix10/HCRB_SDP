function blochvec = StateToBloch(rho)
%STATETOBLOCH Summary of this function goes here
%   Detailed explanation goes here
    d=size(rho,1);
    blochvec=zeros(d^2-1,1);
    basisvec=GeneralizedPauliMatrices(d)./(sqrt(2));
    for i=1:(d^2-1)
        blochvec(i)=trace(rho*basisvec(:,:,i));
    end
end