function [phi, hol] = holopt(n, gamma)


itter      = 100;
nos        = 5;
%n          = 3;
%gamma      = 0.1;
alpha      = [1e-4,1e-4,1e-4];
C    = cell(nos, 1);
C(:) = {n};
state      = cellfun(@(n)randstate(n),C,'un',0);
best_state = {state(1), 1000};
f          = @(x)final_state(alpha, gamma, x, n);
d          = @(x)deriv(alpha, gamma, x, n);
rho_t      = cellfun(f, state,'un',0);
d_rhos     = cellfun(d, state,'un',0);


for i = 1:itter
    var = cellfun(@HolevoCRB_NagSDP, rho_t, d_rhos);
    for j = 1:nos
        if var(j) < 0 
            state{j} = best_state{1};
            var(j)   = best_state{2};
        end
        if var(j) < best_state{2}
            best_state{1} = state{j};
            best_state{2} = var(j);
        end
    end
    state      = genet(state, var,nos, n);
    state{end} = best_state{1};
    rho_t      = cellfun(f, state,'un',0);
    d_rhos     = cellfun(d, state,'un',0);
    best_state{1}
    best_state{2}
end



x0 = [real(best_state{1});imag(best_state{1})];


optphi = fminunc(@(x)func(x,gamma), x0);

if size(optphi,1) == 1
   optphi  = optphi';
end
m   = size(optphi,1);
phi = optphi(1:m/2)+1j*optphi(m/2 +1:m);
phi = phi/sqrt(phi'*phi);

hol = func(optphi);

