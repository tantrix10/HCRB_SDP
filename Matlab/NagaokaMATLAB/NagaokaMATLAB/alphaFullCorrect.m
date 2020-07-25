function alpha = alphaFullCorrect(g,s,d)
%ALPHAMAT Summary of this function goes here
%   Detailed explanation goes here

alpha=blkdiag(1/d,g);
alpha(1,2:end)=s'./sqrt(d);
alpha(2:end,1)=s./sqrt(d);

alpha=blkdiag(1,g);
alpha(1,2:end)=s';
alpha(2:end,1)=s;

end

