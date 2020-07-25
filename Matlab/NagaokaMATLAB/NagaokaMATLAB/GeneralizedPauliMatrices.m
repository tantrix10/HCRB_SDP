function [lambdavec] = GeneralizedPauliMatrices(d,sp)
%GENERALIZEDPAULIMATRICES Gives a vector of generalized Pauli matrices in
%dimension d
%   This functions gives back a 3d array with shape (d^2 -1,d,d). The case of sparse matrices still needs to be implemented 
switch nargin
    case 1
        sp=0;
        lambda=zeros(d,d,d,d);
        for i=0:(d-1)
            for j=0:(d-1)
                lambda(:,:,i+1,j+1)=GenGellMann(i,j,d,sp); % real symmetric
            end
        end
        lambdavec=reshape(lambda,d,d,d^2);
        lambdavec(:,:,1)=[]; %remove the identity
    case 2
        sp=1;
        lambda=cell(d,d);
        for i=0:(d-1)
            for j=0:(d-1)
                lambda{i+1,j+1}=GenGellMann(i,j,d,sp); % real symmetric
            end
        end
end