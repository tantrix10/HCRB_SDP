dd=4;
rnk=2;
%rho=RandomDensityMatrix(dd,0,rnk);
rho = rho;
rhot = (rho+rho')/2
%rho = [0.9 0 0 0.1; 0 0 0 0; 0 0 0 0; 0.1 0 0 0.3 ];
tol=1e-5;
%rho = [8.99999964e-01+0.00000000e+00j 5.36817294e-05+1.07331258e-04j 5.36817294e-05+1.07331258e-04j 2.39999971e-01-9.59839942e-05j;
 %5.36817294e-05-1.07331258e-04j 2.00023994e-08+0.00000000e+00j 1.60019195e-08+0.00000000e+00j 1.78795999e-05-3.57842395e-05j;
 %5.36817294e-05-1.07331258e-04j 1.60019195e-08+0.00000000e+00j 2.00023994e-08+0.00000000e+00j 1.78795999e-05-3.57842395e-05j;
 %2.39999971e-01+9.59839942e-05j 1.78795999e-05+3.57842395e-05j 1.78795999e-05+3.57842395e-05j 9.99999960e-02+0.00000000e+00j]

%random observables
matAre=randn(dd);
matAim=randn(dd);
matBre=randn(dd);
matBim=randn(dd);
%A=matAre+matAre' + 1j.*(matAim-matAim');
%B=matBre+matBre' + 1j.*(matBim-matBim');

%A = [0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0 ];
%B = [0 0 0 -1; 0 0 1 0; 0 1 0 0; -1 0 0 0 ];


%diagonalization
[V,D]=eig(rhot);
[svec,ind] = sort(diag(D),'descend');
V=V(:,ind);
snonzero=svec(svec>tol); %vector containing non-zero eigenvalues

SmatRho=SmatRank(snonzero,dd);

% switch to the eigenbasis
Aeig=V'*A*V;
Beig=V'*B*V;

% definition of the masks to create the vector
maskDiag=diag([true(rnk,1);false(dd-rnk,1)]);
maskRank=[triu(true(rnk),1),false(rnk,dd-rnk);false(dd-rnk,dd)];
maskKern=[false(rnk),true(rnk,dd-rnk);false(dd-rnk,dd)];

% Apply the mask to get the vector
AvecMask=[Aeig(maskDiag);real(Aeig(maskRank));imag(Aeig(maskRank));real(Aeig(maskKern));imag(Aeig(maskKern)) ]
BvecMask=[Beig(maskDiag);real(Beig(maskRank));imag(Beig(maskRank));real(Beig(maskKern));imag(Beig(maskKern)) ]

% Manually create the vector
AeigRnk=Aeig(1:rnk,1:rnk);
BeigRnk=Beig(1:rnk,1:rnk);
Arect=Aeig(1:rnk,rnk+1:rnk+(dd-rnk))
Brect=Beig(1:rnk,rnk+1:rnk+(dd-rnk))
mask=triu(true(rnk),1);
Avec=[diag(AeigRnk);real(AeigRnk(mask));imag(AeigRnk(mask));real(Arect(:));imag(Arect(:)) ]
Bvec=[diag(BeigRnk);real(BeigRnk(mask));imag(BeigRnk(mask));real(Brect(:));imag(Brect(:)) ]

% check both ways give the correct result
Avec'*SmatRho*Bvec - trace(rhot*A*B)
AvecMask'*SmatRho*BvecMask - trace(rhot*A*B)












