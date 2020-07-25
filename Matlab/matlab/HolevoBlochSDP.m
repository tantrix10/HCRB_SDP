function [holCRB,status] = HolevoBlochSDP(rho,drhovec)
%PURENAGAOKAHOLEVOCRB Summary of this function goes here
%   Detailed explanation goes here

d     = size(rho,1);
alpha = [1e-4,1e-4,1e-4];
gamma = 0;
n     = 2;
rho = final_state(alpha, gamma, phi, n);


npar=size(drhovec,3); %drhovec is assumed to be an array of size (d,d,npar)
W=eye(npar);


s=StateToBloch(rho);

dsvec=zeros(d^2-1,npar); %the columns are the derivatives of the Bloch vector, 
                         %this is done so that I can multiply the matrix from left to act collectively on the vectors
                         %this is basically the Jacobian matrix of the vector s
                         
for i=1:npar %the code is probably slow because I'm not vectorializing anything
    dsvec(:,i)=StateToBloch(drhovec(:,:,i)); %maybe it might be worth to used a saved vector of Gell-Mann matrices for higher dimension?
end

S=Smat(rho)-kron(s,s'); %the matrix F seems to be transposed wrt to the one I got in mathematica for Qubits -> CHECK!!
R=chol(S);

sqrtW=W^(1/2);

cvx_begin sdp
    variable V(npar,npar) symmetric
    variable X(d^2-1,npar)
    minimize trace(sqrtW*V*sqrtW)
    subject to
        [ V , X'*R' ; R*X , eye(d^2-1)] >= 0
        dsvec'*X==eye(npar)
cvx_end

holCRB=cvx_optval;
status=cvx_status;

end