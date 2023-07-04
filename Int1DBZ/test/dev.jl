# main development and testing area for Int1DBZ ideas.
# Barnett June-July 2023
using Int1DBZ
using Printf
using OffsetArrays
using LinearAlgebra
using Random.Random
using TimerOutputs
using Gnuplot

M=20            # max mag Fourier freq index (200 to make fevals slow)
η=1e-6; ω=0.5; tol=1e-7;  # 1e-8 too much for M=200 realadap to handle :(
verb = 0
Random.seed!(0)         # set up 1D BZ h(x) for denominator
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # F-coeffs of h(x)
hm = (hm + conj(reverse(hm)))/2                     # make h(x) real for x Re

# benchmark various 1D BZ quadr methods...
TIME = TimerOutput()
@printf "\nConventional quadrature via QuadGK:\n"
@printf "test realadap for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Aa = realadap(hm,ω,η,tol=tol, verb=1)
@printf "\tAa = "; println(Aa)
TIME(realadap)(hm,ω,η,tol=tol)
@printf "test realadap_lxvm for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Al = realadap_lxvm(hm,ω,η,tol=tol, verb=1)
@printf "\tAl = "; println(Al)
TIME(realadap_lxvm)(hm,ω,η,tol=tol)
Am, E, segs, numevals = realmyadap(hm,ω,η,tol=tol)
if (verb>0)
    plot(segs,:realmyadap)  # show adaptivity and roots of denom...
    xr = shifted_fourier_series_roots(hm,ω,η);
    @gp :realmyadap :- real(xr) imag(xr) "w p pt 2 lc rgb 'red' t 'roots'"
end
@printf "test realmyadap (same pars): fevals=%d, nsegs=%d, claimed err=%g\n" numevals length(segs) E
TIME(realmyadap)(hm,ω,η,tol=tol)    # timing valid since func not passed in :)
rho0=1.0    # for readquadinv; gets slower either side
Ap, E, segs, numevals = realquadinv(hm,ω,η,tol=tol,rho=rho0)
@printf "test realquadinv (same pars): fevals=%d, nsegs=%d, claimed err=%g\n" numevals length(segs) E
@printf "\tAp = "; println(Ap)
@printf "\t\tabs(Ap-Aa)=%.3g\n" abs(Ap-Aa)
TIME(realquadinv)(hm,ω,η,tol=tol,rho=rho0)
print_timer(TIME, sortby=:firstexec)   # otherwise randomizes order!
if (verb>0)
    plot(segs,:realquadinv)
    @gp :realquadinv :- real(xr) imag(xr) "w p pt 2 lc rgb 'red' t 'roots'"
end
#Gnuplot.quitall()

# examine segs eval'ed vs chosen: use to tweak rho
#Ap, E, segs, numevals = realquadinv(hm,ω,η,tol=tol,rho=1.0); abs(Ap-Aa), Int(numevals/15), length(segs)

#=
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1
ab=[3.0,3.4];   # key region where rho=0.5 caused splitting
Aa,E,segs,numevals = realmyadap(hm,ω,η; tol=tol, ab=ab);
Ar,E,segsr,numevals = realquadinv(hm,ω,η; tol=tol, ab=ab);
@printf "realmyadap %d segs vs realquadinv %d segs\n" length(segs) length(segsr)
@printf "smaller-ab check abs(Ar-Aa)=%.3g\n" abs(Ar-Aa)
@btime realmyadap(hm,ω,η; tol=tol, ab=ab);
@btime realquadinv(hm,ω,η; tol=tol, ab=ab);
# tol=1e-10 we destroy realmyadap(miniquadgk)... but that's too accurate
=#



