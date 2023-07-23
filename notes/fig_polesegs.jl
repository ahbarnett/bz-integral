# a fig for poles subtraction note. Barnett 7/21/23
# run with:
#  from Int1DBZ:   julia -t1 --project=.
#  julia> include("../notes/fig_polesegs.jl")

using Int1DBZ
using Printf
using OffsetArrays
using StaticArrays
using Random.Random
using Gnuplot        # crappy for now

n=8             # matrix size for H (1:scalar)
M=10            # max mag Fourier freq index (eg 200 to make fevals slow)
η=1e-5; ω=0.5; tol=1e-6; mtail = 1e-2;

Random.seed!(0)         # set up 1D BZ h(x) for denominator
if false && n==1        # [obsolete] scalar case without SMatrix-valued coeffs
    Hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)  # F-coeffs of h(x)
    Hm = (Hm + conj(reverse(Hm)))/2                 # make h(x) real for x Re
else
    mlist = -M:M  # matrix, OV of SA's version, some painful iterators here...
    decayrate = log(1/mtail)/M
    am = OffsetVector(exp.(-decayrate*abs.(mlist)), mlist)  # rand w/ decay
    Hm = OffsetVector([SMatrix{n,n}(am[m] * randn(ComplexF64,(n,n))) for m in mlist], mlist)
     Hmconj = OffsetVector([Hm[m]' for m in mlist], mlist)
    Hm = (Hm + reverse(Hmconj))/2                     # H(x) hermitian if x Re
end
@printf "Test n=%d M=%d ω=%g η=%g tol=%g...\n" n M ω η tol
xr = BZ_denominator_roots(Hm,ω,η);     # count roots (NEVs)
@printf "global %d roots (or NEVs) of which %d η-near Re axis.\n" length(xr) sum(abs.(imag.(xr)).<10η)

#Aa = realadap(Hm,ω,η,tol=tol, verb=1)
#@printf "\trealadap integral Aa = "; println(Aa)
Al = realadap_lxvm(Hm,ω,η,tol=tol)
@printf "\trealadap_lxvm ans Al = "; println(Al)
Am, E, segs, numevals = realmyadap(Hm,ω,η,tol=tol, verb=1)
@printf "\tabs(Am-Al)=%.3g\n" abs(Am-Al)

@gp real(xr) imag(xr) "w p pt 2 lc rgb 'red' t 'poles'"
plot!(segs)
@gp :- yrange=[-.5,.5] xlabel="Re k" ylabel="Im k"
@gp :- title="(a) plain adaptive GK"
save(term="epscairo size 7,2", output="segsa.eps")    # inches

rho0=1.0         # for readquadinv
rmeth="PR"        # root-finding meth
Ap, E, segs, numevals = realquadinv(Hm,ω,η,tol=tol,rho=rho0,rootmeth=rmeth,verb=1)
@printf "\tabs(Ap-Al)=%.3g\n" abs(Ap-Al)
@gp real(xr) imag(xr) "w p pt 2 lc rgb 'red' t 'poles'"
plot!(segs)
@gp :- yrange=[-.5,.5]  xlabel="Re k" ylabel="Im k"
@gp :- title="(b) pole-subtracting adaptive GK"
save(term="epscairo", output="segsb.eps")   # size seems to be remembered

# sys call via backquotes: note doesn't like wildcards (OS-dependent)...
run(`mv segsa.eps segsb.eps ../notes/`)

#Gnuplot.quitall()
