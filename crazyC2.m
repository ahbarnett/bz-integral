% test crazy C2 idea for Brillouin zone (BZ) integration over [0,2pi)^2,
% const-shift in C2 for x,y.  Fails.
% Barnett 3/18/22
clear; close all; verb=1;

nmax = 1;          % max Fourier freq in each dim
co = energybandcoeffs2d(nmax);
f = @(x,y) analenergyband2d(x,y,co);   % band func over BZ: (x,y) in [0,2pi)^2
                                       % (analytic in x and y version)

eta = 1e-1;   % imag shift param (inverse hardness of problem)
gf = @(x,y) 1 ./ (1i*eta + f(x,y));     % "Green's func" or something, complex

if verb>1   % conv plot
ns=300:100:1000;    % plain double PTR : note n=1e3 needed for 1e-10 @ eta=.1
for i=1:1:numel(ns), nx=ns(i);
  g=(1:nx)/nx*2*pi; [xx yy] = ndgrid(g,g);
  h = g(2)-g(1);
  ggf = gf(xx,yy);       % integrand samples (ok since don't vary eta)
  I(i) = h*h*sum(ggf(:));
  fprintf('nx=%d:  \tI=%.12g +\t%.12gi\n',nx,real(I(i)),imag(I(i)))
end
Iex = I(end);
figure(1); semilogy(ns,abs(I-Iex)/abs(Iex),'+-');
xlabel('nx'); ylabel('rel err'); title('double-PTR conv naive')
end

nx = 400;
g=(1:nx)/nx*2*pi; [xx yy] = ndgrid(g,g);
ggf = gf(xx,yy);

if verb
figure(2);
subplot(1,2,1); imagesc(g,g,real(ggf).')
xlabel('x'); ylabel('y'); axis equal xy tight; colorbar;
title(sprintf('Re (f-i\\eta)^{-1} for \\eta=%.3g',eta));
subplot(1,2,2); imagesc(g,g,imag(ggf).')
xlabel('x'); ylabel('y'); axis equal xy tight; colorbar;
title(sprintf('Im (f-i\\eta)^{-1} for \\eta=%.3g',eta));
end

figure(3);  % anim in complex x plane of a const-y slice.
ss = -2:0.02:2;        % im(x) to test
[rxx ixx] = ndgrid(g,ss); xx = rxx+1i*ixx;
for yc = 0:0.03:2*pi;
  yy = yc + 0*xx;   % const slice
  ggf = gf(xx,yy).';   % ss down, by xx across
  imagesc(g,ss,real(ggf))
  caxis(10*[-1 1])
  xlabel('Re x'); ylabel('Im x'); axis equal tight; colorbar
  title(sprintf('Re (f-i\\eta)^{-1}, x in C, slice real y=%.3g',yc));
  drawnow
end
%return

figure(4);
nx = 400;
g=(1:nx)/nx*2*pi; [xx yy] = ndgrid(g,g);
for s = 0:0.01:1
  ggf = gf(xx+1i*s,yy);       % integrand samples eval @ imag-shifted coords
  %  ggf = gf(xx,yy+1i*s);       % integrand samples eval @ imag-shifted coords
  imagesc(g,g,real(ggf).')
  caxis(20*[-1 1])
  xlabel('x'); ylabel('y'); axis equal xy tight; colorbar;
  title(sprintf('Re (f-i\\eta)^{-1}, s=%.3g',s));
  drawnow
end
