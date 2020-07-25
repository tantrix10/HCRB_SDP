function [minCRB,Uopt,status] = optCRBproj(rho,drhovec,W)
%OPTCRBPROJ Summary of this function goes here
%   Detailed explanation goes here

d=size(rho,1);
npar=size(drhovec,3);

gell=GeneralizedPauliMatrices(d);

fobj = @(x)real(objFun(x,gell,rho,drhovec,W,d,npar));

% options = optimoptions(@patternsearch,'Algorithm','interior-point','MaxFunctionEvaluations',100000);

x0=randn(d^2,1);
% x0=[zeros(d^2-1,1) ; 1];
% [Uopt,minCRB,status] = patternsearch(fobj,x0,[],[],[],[],[],[],[],options);
[Uopt,minCRB,status] = patternsearch(fobj,x0);
% [Uopt,minCRB,status] = fminunc(fobj,x0);
% [Uopt,minCRB,status] = simulannealbnd(fobj,x0);
% [Uopt,minCRB,status] = ga(fobj,d^2);
% [Uopt,minCRB,status] = particleswarm(fobj,d^2);

% problem = createOptimProblem('fmincon','objective',fobj,'x0',x0);
% gs = GlobalSearch;
% [Uopt,minCRB,status,outptg,manyminsg] = run(gs,problem);

end