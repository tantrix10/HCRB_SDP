function[drhos] = deriv(alpha, gamma, phi, n)

delta = 1e-8;

drhos(:,:,1) = (final_state(alpha + [delta,0,0], gamma, phi, n)-final_state(alpha, gamma, phi, n))/delta;
drhos(:,:,2) = (final_state(alpha + [0,delta,0], gamma, phi, n)-final_state(alpha, gamma, phi, n))/delta
drhos(:,:,3) = (final_state(alpha + [0,0,delta], gamma, phi, n)-final_state(alpha, gamma, phi, n))/delta
end