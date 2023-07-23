# main development and testing area for Int1DBZ ideas.
# Barnett June-July 2023. 7/4/23 matrix case, also works for n=1 scalar.
using Int1DBZ
using Printf
using OffsetArrays
using StaticArrays
using LinearAlgebra
BLAS.set_num_threads(1)
using Random.Random
using TimerOutputs
using Gnuplot

n=8             # matrix size for H (1:scalar)
M=10            # max mag Fourier freq index (eg 200 to make fevals slow)
η=1e-5; ω=0.5; tol=1e-6; mtail = 1e-2;
verb = 0
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

# for use w/ julia -t1 --track-allocation=user --project=.
#Ap, E, segs, numevals = realquadinv(Hm,ω,η,tol=tol,rho=0.8)
#Am, E, segs, numevals = realmyadap(Hm,ω,η,tol=tol);
#stop

# benchmark various 1D BZ quadr methods...
TIME = TimerOutput()
#Aa = realadap(Hm,ω,η,tol=tol, verb=1)
#@printf "\trealadap integral Aa = "; println(Aa)
#TIME(realadap)(Hm,ω,η,tol=tol)
Al = realadap_lxvm(Hm,ω,η,tol=tol)
TIME(realadap_lxvm)(Hm,ω,η,tol=tol)    # TimerOutputs no $Hm interpolation :(
@printf "\trealadap_lxvm ans Al = "; println(Al)
Am, E, segs, numevals = realmyadap(Hm,ω,η,tol=tol, verb=1)
@printf "\tabs(Am-Al)=%.3g\n" abs(Am-Al)
TIME(realmyadap)(Hm,ω,η,tol=tol)    # (timing valid since func not passed in :)
if (verb>0)          # show adaptivity around poles (roots of denom)
    @gp :realmyadap real(xr) imag(xr) "w p pt 2 lc rgb 'red' t 'poles'"
    plot!(segs,:realmyadap)
end
rho0=1.0  #0.8    # for readquadinv; gets slower either side
rmeth="PR"
Ap, E, segs, numevals = realquadinv(Hm,ω,η,tol=tol,rho=rho0,rootmeth=rmeth,verb=1)
@printf "\tabs(Ap-Al)=%.3g\n" abs(Ap-Al)
TIME(realquadinv)(Hm,ω,η,tol=tol,rho=rho0,rootmeth=rmeth)
print_timer(TIME, sortby=:firstexec)   # otherwise randomizes order!
if (verb>0)
    @gp :realquadinv real(xr) imag(xr) "w p pt 2 lc rgb 'red' t 'poles'"
    plot!(segs,:realquadinv)
end
#Gnuplot.quitall()

#=
using Profile
@profile (for i=1:200;
          Ap, E, segs, numevals = realquadinv(Hm,ω,η,tol=tol,rho=rho0); end)
Profile.print()
=#
