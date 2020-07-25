function [Smat] = SmatRank(snonzero,d)
%SMATRANK Summary of this function goes here
%   Detailed explanation goes here

rnk=size(snonzero,1);

dim=2*rnk*d - rnk^2;

mask=triu(true(rnk),1);  %select only the elements in the upper triangular part of the rank (without diagonal)

scols=kron(ones(rnk,1),snonzero');
srows=scols';
siminsj=scols-srows;
siplsj=srows+scols;

diagS=[snonzero; siplsj(mask); siplsj(mask); repmat(snonzero,2*(d-rnk),1)];

Smat=spdiags(diagS,0,dim,dim);

if rnk~=1
    offdRank=1j.*spdiags(siminsj(mask),0,(rnk^2-rnk)/2,(rnk^2-rnk)/2);
else
    offdRank=0;
end

Smat(rnk+(rnk^2-rnk)/2+1:rnk+(rnk^2-rnk),rnk+1:rnk+(rnk^2-rnk)/2)=-offdRank;
Smat(rnk+1:rnk+(rnk^2-rnk)/2,rnk+(rnk^2-rnk)/2+1:rnk+(rnk^2-rnk))=offdRank;

offdKer=-1j.*spdiags(repmat(snonzero,(d-rnk),1),0,rnk*(d-rnk),rnk*(d-rnk));
% sparse(1j.*diag(repmat(snonzero,(d-rnk),1)));

Smat(rnk+(rnk^2-rnk)+rnk*(d-rnk)+1:rnk+(rnk^2-rnk)+2*rnk*(d-rnk),rnk+(rnk^2-rnk)+1:rnk+(rnk^2-rnk)+rnk*(d-rnk))=-offdKer;
Smat(rnk+(rnk^2-rnk)+1:rnk+(rnk^2-rnk)+rnk*(d-rnk),rnk+(rnk^2-rnk)+rnk*(d-rnk)+1:rnk+(rnk^2-rnk)+2*rnk*(d-rnk))=offdKer;


end

