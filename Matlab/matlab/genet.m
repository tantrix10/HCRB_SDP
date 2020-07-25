function[new_states] = genet(statesin, variences,nos,n)
cost      = 1./variences;
costsum   = sum(cost);

mating_pool_relative = round((cost./costsum).*100);
mating_pool          = [];
new_states           = [];
for i = 1:size(mating_pool_relative,2)
    mating_pool = [mating_pool, repmat(i,1,mating_pool_relative(i))];
end


for j = 1:nos
    state1 = statesin{randsample(mating_pool,1)};
    state2 = statesin{randsample(mating_pool,1)};
    splice = randsample([1:2^n],1);
    n_state = [state1(1:splice);state2(splice+1:end)];
    n_state = mutate(n_state);
    norm = sqrt(n_state'*n_state);
    if norm < 1e-10
        n_state = randstate(n);
    else 
        n_state = n_state/norm;
    end
    new_states{end+1} = n_state;
end
