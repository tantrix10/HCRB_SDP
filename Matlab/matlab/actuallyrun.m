gammas = 0:0.1:0.7;
n = 2;
len = size(gammas,2);
outputs4 = cell(len,1);

for i = 1:len
    outputs4{i} = holopt(n,gammas(i));
    save('outputs4')
end

n = 5;
outputs5 = cell(len,1);

for i = 1:len
    outputs5{i} = holopt(n,gammas(i));
    save('outputs5')
end