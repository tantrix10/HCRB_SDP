function [state] = randstate(n)
state = rand(2^n,1) + 1j*rand(2^n,1);
state = state/(state'*state);