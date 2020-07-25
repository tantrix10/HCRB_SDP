psi=[    0.545828930000000 - 0.449714490000000i; -0.517517880000000 + 0.248561720000000i ; 0.390972750000000 - 0.131758420000000j; 0.00000000000000 + 0.00000000000000j];
% psi=[0.949 ;0.;0.;0.316];
% psi=psi/sqrt(real(psi'*psi));
alpha=repmat(1e-4,[1,3]);
gamma=.2;
n=2;
W=eye(3);
rho = final_state( alpha, gamma,psi,n);
% rho(logical(eye(4)))=real(diag(rho)) %makes the diagonal elements
% strictly real

drhos = deriv(alpha, gamma, psi, n);
drhosimple = deriv_simple(alpha, gamma, psi, n);

drhopython=zeros(2^n,2^n,3);
drhopython(:,:,1)=cell2mat(struct2cell(load('drho1.mat')));
drhopython(:,:,2)=cell2mat(struct2cell(load('drho2.mat')));
drhopython(:,:,3)=cell2mat(struct2cell(load('drho3.mat')));

[holCRBcvxSDPpython,~] = HolevoCRB_NagSDP(rho,drhopython,W)
[holCRBcvxSDP,~] = HolevoCRB_NagSDP(rho,drhos,W)

