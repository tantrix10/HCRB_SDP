function [state] = mutate(statein)
d = size(statein, 1);
state = statein;
for i = 1:d
    chance = rand;
    if chance < 1/d
        mut = randsample([1:3],1);
        if mut == 1 
            state(i) = 2*rand -1 +1j*(2*rand -1);
        elseif mut == 2 
            state(i) = state(randsample(d,1));
        else
            state(i) = 0;
        end
    end
end