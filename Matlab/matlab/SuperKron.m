function [ H ] = SuperKron( y, i )
%SINGLE_PARTICLE_HAMILTONIAN Summary of this function goes here
%   Detailed explanation goes here
if (i)==2
    H=kron(y{1},y{2});
else
    H=kron(SuperKron(y,i-1),y{i});
end