function [f fc] = analenergyband2d(x,y,co)
% ANALENERGYBAND2D  energy f vs (x,y) wavevector in [0,2pi)^2, 2D Fourier series
%
% [f fcount] = analenergyband2d(x,y,co) returns f(x,y) values of same shape as x
%  and y arrays, vectorized. co is a 2D Fourier coefficient array of odd size
%  with freqs -nmax:nmax. fcount is a counter increasing one per func eval.
%  It is analytic w.r.t. x and y.

% Barnett 12/9/12, changed for complex coord inputs 3/18/22.
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
% (do not take real part)
