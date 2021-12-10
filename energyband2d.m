function [f fc] = energyband2d(x,y,co)
% ENERGYBAND2D  energy f vs (x,y) wavevector in [0,2pi)^2, 2D Fourier series
%
% [f fcount] = energyband2d(x,y,co) returns f(x,y) values of same shape as x
%  and y arrays, vectorized. co is a 2D Fourier coefficient array of odd size
%  with freqs -nmax:nmax. fcount is a counter increasing one per func eval.

% Barnett 12/9/12
persistent fcount
if isempty(fcount), fcount=1; else fcount=fcount+1; end, fc=fcount;

f = 0*x;
N = (size(co,1)-1)/2;    % max freq per dim
for n=-N:N
    for m=-N:N
        a = n*x + m*y;
        f = f + co(n+N+1,m+N+1) * (cos(a) + 1i*sin(a));
    end
end
f = real(f);   % *** this means the above is twice as slow as could be
