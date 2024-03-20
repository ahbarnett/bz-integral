# try combining quadratic pade with GCQ for auto-sqrt handling on 1 seg
# Barnett 2/22/24, restart post-Bonaire 3/20/24
using Int1DBZ
using Printf
using GLMakie
include("../src/genchebquad.jl")

# integrand f and answer I:  2D tight-binding case 
om = 0.6      # overall energy
eta = 1e-2    # broadening (think of as imag part of om)
G1(om) = 2pi/(1im*sqrt(1-om^2))    # x-integral done, 1D tight-binding model
f(y) = G1(om + 1im*eta - cos(y))   # integrand for middle integral
K(k) = Int1DBZ.ellipkAGM(k)               # local code for complete elliptic integral
G2(om) = 4pi*(K(om/2) - 1im*K(sqrt(1-(om/2)^2)))     # 2D TB model
Ie = G2(om + 1im*eta)                 # analytic (exact) answer
@printf "2D tight-binding test: om=%g, eta=%g\n" om eta

# test conventional adap
Ia,E,segs,nev = miniquadgk(f,0,2pi,rtol=1e-10)
@printf "exact:\tIe = %.12g + %.12gi\n" real(Ie) imag(Ie)
@printf "adap:\tIa = %.12g + %.12gi\t (relerr=%.3g, esterr=%.3g, nsegs=%d, nev=%d)\n" real(Ia) imag(Ia) abs(Ia-Ie)/abs(Ie) E length(segs) nev

# test by-hand composite quad, uniform segments over [0,2pi)
ns = 100
L = 2pi/ns
r = gkrule(); p = length(r.x)
Iu = 0.0; Eu = 0.0;
for i=1:ns
    s = applygkrule(f,(i-1)*L,i*L,r)
    Iu += s.I; Eu += s.E
end
@printf "unif:\tIu = %.12g + %.12gi\t (relerr=%.3g, esterr=%.3g, nsegs=%d, nev=%d)\n" real(Iu) imag(Iu) abs(Iu-Ie)/abs(Ie) Eu ns ns*p
