function h = NagaokaObjFun(X,W,F,d,npar)
%PURENAGAOKAOBJFUN Summary of this function goes here
%   Detailed explanation goes here

% d=size(X,1)/2;
X=reshape(X,[d,npar]);

ReV=X'*X;
ImV=X'*F*X;

h= trace(W*ReV) + sum(abs( eig( W*ImV ) ));

end