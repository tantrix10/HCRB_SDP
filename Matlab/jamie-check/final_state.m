function[rhon] = final_state( alpha, gamma,phi,n)


rho = phi*phi';
sz = [1  0 ; 0 -1];
sy = [0 -1j; 1j 0];
sx = [0 , 1; 1  0];
Hx=zeros(2^n);

Hy=zeros(2^n);
Hz=zeros(2^n);
h=cell(1,n);

for j =1:n
    h{j}=eye(2);
end

for i=1:n
    x=h;
    x{i}=sx;
    Hx=Hx + SuperKron(x,n);
end

for i=1:n
    y=h;
    y{i}=sy;
    Hy=Hy + SuperKron(y,n);
end

for i=1:n
    z=h;
    z{i}=sz;
    Hz=Hz + SuperKron(z,n);
end


H = alpha(1)*Hx+alpha(2)*Hy+alpha(3)*Hz;


U = expm(-1j*H);

rhop = U*rho*U';

e=npermutek(['b','c'],n);


c = [1 0; 0 sqrt(1-gamma)]; %#ok<NASGU>
b = [0 0; 0 sqrt(gamma)  ]; %#ok<NASGU>


ek=cell((2^n),n);

for i=1:2^n
    for j=1:n
        ek{i,j}=eval(e(i,j));
    end
end

E=cell(1,2^n);

for i=1:(2^n)
    E{i}=kronr(ek(i,:),n);
end

rhon=zeros(2^n);

for i =1:2^n
    rhon=rhon+(E{i}*rhop*E{i});

end

