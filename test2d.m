% demo script for Brillouin zone (BZ) integration over [0,2pi)^2
% Barnett 12/9/21, from Jason Kaye's talk.
% Firstly comparing 2d iterated vs 2d tiled, dependence on eta.
close; verb = 1;  % verbosity: 0=final plot only

nmax = 1;          % max Fourier freq in each dim
co = energybandcoeffs2d(nmax);
f = @(x,y) energyband2d(x,y,co);   % band func over BZ: (x,y) in [0,2pi)^2

eta = 1e-1;   % imag shift param (inverse hardness of problem)
g = @(x,y) 1 ./ (1i*eta + f(x,y));     % "Green's func" or something, complex

if verb, figure(1); clf; nx=500; gg=(1:nx)/nx*2*pi; [xx yy] = ndgrid(gg,gg);
    ff = f(xx,yy); subplot(1,2,1); surf(xx,yy,ff); shading interp; axis vis3d
    hold on; surf(xx,yy,0*xx,0*xx);  % use color=0 for f=0 plane
    xlabel('x'); ylabel('y'); title('f(x,y) energy surface & zero slice');
    subplot(1,2,2); I = abs(1./(1i*eta + ff)); imagesc(gg,gg,log10(I.'));
    xlabel('x'); ylabel('y'); axis equal xy tight; colorbar;
    title(sprintf('log_{10} |g(x,y)| for \\eta=%.3g',eta));
    set(gcf,'paperposition',[0 0 8 8]); print -dpng figs/test2d_bz.png
end

tol = 1e-3;
fmax = 1e6;  % max # evals (quad2d only)

If = quad2d(f,0,2*pi,0,2*pi,'abstol',tol);   % sanity check
fprintf('sanity: err integr plain f = %.3g\n', If - (2*pi)^2*co(nmax+1,nmax+1))

fprintf('single eta=%.3g...\n',eta)
tic; [~,i]=f(0,0);          % stupid way to access feval counter
I2 = quad2d(g,0,2*pi,0,2*pi,'abstol',tol,'maxfunevals',fmax); % quad2d=tiled
t2=toc; [~,j]=f(0,0); nf2=j-i-1;
fprintf('2d-adaptive tiled:   \tRe(I)=%.10g \t #f=%d\n',real(I2),nf2)
tic; [~,i]=f(0,0);
I2i = integral2(g,0,2*pi,0,2*pi,'abstol',tol,'method','iterated');
t2i=toc; [~,j]=f(0,0); nf2i=j-i-1;
fprintf('2d-adaptive iterated:\tRe(I)=%.10g \t #f=%d\n',real(I2i),nf2i)

ne = 12;       % how many eta vals
etas = 10.^linspace(0,-3.5,ne);    % loop over eta (-3.5 is the worst)
nfs = nan(ne,2); Is=nfs;              % # f evals, I vals: each col a method
for e=1:ne
    eta = etas(e); fprintf('eta=%.3g ...\n',eta);
    g = @(x,y) 1 ./ (1i*eta + f(x,y));     % update our integrand
    [~,i]=f(0,0);
    Is(e,1) = quad2d(g,0,2*pi,0,2*pi,'abstol',tol,'maxfunevals',fmax);
    [~,j]=f(0,0); nfs(e,1)=j-i-1;
    fprintf(' 2d-adaptive tiled:   \tRe(I)=%.10g \t #f=%d\n',real(Is(e,1)),nfs(e,1))
    [~,i]=f(0,0);
    Is(e,2) = integral2(g,0,2*pi,0,2*pi,'abstol',tol,'method','iterated');
    [~,j]=f(0,0); nfs(e,2)=j-i-1;
    fprintf(' 2d-adaptive iterated:\tRe(I)=%.10g \t #f=%d\n',real(Is(e,2)),nfs(e,2))
end

figure(2); clf; subplot(1,2,1); plot(etas,Is,'+-'); xlabel('\eta'); ylabel('I');
title('I vs \eta approaching limit?')
subplot(1,2,2); loglog(etas,nfs,'+-'); xlabel('\eta'); ylabel('func evals');
axis tight; title('effort scaling')
hold on; plot(etas,20*log(1./etas)./etas,'g--',  etas, 1e2*log(1./etas).^2, 'r--');
legend('quad2d tiled','integral2 iter','\eta^{-1} log \eta^{-1}','log^2 \eta^{-1}');
set(gcf,'paperposition',[0 0 8 8]); print -dpng figs/test2d_etascal.png

