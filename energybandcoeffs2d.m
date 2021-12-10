function co = energybandcoeffs2d(nmax)
% ENERGYBANDCOEFFS2D   choose some complex double Fourier series coeffs
%
% co = energybandcoeffs2d(nmax)
%  chooses random complex 2D Fourier coeffs for a toy band surface of
%  frequencies up to nmax.
    
% Barnett 12/9/21
N = 2*nmax+1;    % # terms per dim
rng(3);          % rng(3) for nmax=1 gives nice simple loop level curve
co = randn(N,N) + 1i*randn(N,N);        % freqs are -nmax:nmax
co = co + flipud(fliplr(conj(co)));       % Hermitian symm
