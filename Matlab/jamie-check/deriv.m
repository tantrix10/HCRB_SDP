function[drhos] = deriv(alpha, gamma, phi, n)

delta = 1e-8;
temp_alpha1    = alpha;
temp_alpha2    = alpha;
temp_alpha3    = alpha;
temp_alpha4    = alpha;
temp_alpha1(1) = temp_alpha1(1) + 2*delta;
temp_alpha2(1) = temp_alpha2(1) + delta;
temp_alpha3(1) = temp_alpha3(1) - delta;
temp_alpha4(1) = temp_alpha4(1) - 2*delta;
t1             = final_state(temp_alpha1, gamma, phi, n);
t2             = final_state(temp_alpha2, gamma, phi, n);
t3             = final_state(temp_alpha3, gamma, phi, n);
t4             = final_state(temp_alpha4, gamma, phi, n);
out1           = (-t1 + (8*t2) - (8*t3) + t4)/(12*delta);

temp_alpha1    = alpha;
temp_alpha2    = alpha;
temp_alpha3    = alpha;
temp_alpha4    = alpha;
temp_alpha1(2) = temp_alpha1(2) + 2*delta;
temp_alpha2(2) = temp_alpha2(2) + delta;
temp_alpha3(2) = temp_alpha3(2) - delta;
temp_alpha4(2) = temp_alpha4(2) - 2*delta;
t1             = final_state(temp_alpha1, gamma, phi, n);
t2             = final_state(temp_alpha2, gamma, phi, n);
t3             = final_state(temp_alpha3, gamma, phi, n);
t4             = final_state(temp_alpha4, gamma, phi, n);
out2           = (-t1 + (8*t2) - (8*t3) + t4)/(12*delta);


temp_alpha1    = alpha;
temp_alpha2    = alpha;
temp_alpha3    = alpha;
temp_alpha4    = alpha;
temp_alpha1(3) = temp_alpha1(3) + 2*delta;
temp_alpha2(3) = temp_alpha2(3) + delta;
temp_alpha3(3) = temp_alpha3(3) - delta;
temp_alpha4(3) = temp_alpha4(3) - 2*delta;
t1             = final_state(temp_alpha1, gamma, phi, n);
t2             = final_state(temp_alpha2, gamma, phi, n);
t3             = final_state(temp_alpha3, gamma, phi, n);
t4             = final_state(temp_alpha4, gamma, phi, n);
out3           = (-t1 + (8*t2) - (8*t3) + t4)/(12*delta);

drhos = zeros(2^n, 2^n, 3);
drhos(:,:,1) = out1;
drhos(:,:,2) = out2;
drhos(:,:,3) = out3;
end