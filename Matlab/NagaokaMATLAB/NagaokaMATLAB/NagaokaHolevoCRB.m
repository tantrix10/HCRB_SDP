function [holCRB,Xopt,FmatfullONbasis,exitflag] = NagaokaHolevoCRB(rho,drhovec,W,tol)
%PURENAGAOKAHOLEVOCRB Summary of this function goes here
%   Detailed explanation goes here

if nargin<4
    tol=1.0e-8; %tolerance in considering the eigenvalues of the state zero
end

d=size(rho,1);
npar=size(drhovec,3); %drhovec is assumed to be an array of size (d,d,npar)
if nargin==2
    W=eye(npar);
end

s=StateToBloch(rho);

dsvec=zeros(d^2-1,npar); %the columns are the derivatives of the Bloch vector, 
                         %this is done so that I can multiply the matrix from left to act collectively on the vectors
                         %this is basically the Jacobian matrix of the vector s
                         
for i=1:npar %the code is probably slow because I'm not vectorializing anything
    dsvec(:,i)=StateToBloch(drhovec(:,:,i)); %maybe it might be worth to used a saved vector of Gell-Mann matrices for higher dimension?
end

[Gmat,Fmat]=GFmats(rho); %the matrix F seems to be transposed wrt to the one I got in mathematica for Qubits -> CHECK!!
Qmat=Gmat-kron(s,s') %Gmat e Qmat sono singular se lo stato non è full rank, also alpha...

lvec=zeros(d^2,npar);
if det(Qmat) > tol
    lvec(2:d^2,:)=Qmat*dsvec; %check if this works all-right even when the state is not full rank!
    lvec(1,:)=-s'*lvec(2:d^2,:);
else
    Lops=SLDsEigen(rho,drhovec,tol);
    for i=1:npar %the code is probably slow because I'm not vectorializing anything
        lvec(2:d^2,i)=StateToBloch(Lops(:,:,i)); 
        lvec(1,i)=trace((eye(d)/d)*Lops(:,:,i));
    end
end

% Qbasis=orthogonalize(alphaFull(Gmat,s),eye(d^2)); %this matrix has the orthonormal vectors as COLUMNS, the normalization does not matter
alf=alphaFullCorrect(Gmat,s,d);
[Qbasis,eigAlf] = eig(alf); %this matrix has the orthonormal vectors as COLUMNS
Qbasis=Qbasis/(eigAlf^(0.5)); %in this way they are normalized wrt the scalar product defined by the matrix alf

MMat=real(Qbasis\lvec);

Fmatfull=[zeros(1,d^2); zeros(d^2-1,1) , Fmat];
FmatfullONbasis=real(Qbasis'*Fmatfull*Qbasis);

Acon=kron(eye(npar),MMat');
bcon=eye(npar);
bcon=bcon(:);

dim=d^2;

fNag = @(x)real(NagaokaObjFun(x,W,FmatfullONbasis,dim,npar));
x0=rand(dim*npar,1);

% x0=[1 ,5 ; 2, 6, ; 3, 7 ; 4,8];
% x0compl=[1 + 1j*3 ,5 + 1j*7 ; 2 + 1j*4,6+ 1j*8];

% options = optimoptions('fmincon');
options = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunctionEvaluations',100000);
[Xopt,holCRB,exitflag] = fmincon(fNag,x0,[],[],Acon,bcon,[],[],[],options);

Xopt=reshape(Xopt,[dim,npar]);

end