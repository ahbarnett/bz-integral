# tester for contour 1D BZ module in this dir. Barnett 12/19/22

push!(LOAD_PATH,".")
using Cont1DBZ
using LinearAlgebra
using Printf
using OffsetArrays

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1   # for btime only
using Test

# plotting
using Gnuplot
using ColorSchemes
Gnuplot.options.term="qt 0 font \"Sans,9\" size 1000,500"    # window pixels

dotime = 0

# -------- module method tests ----------
# test eval h...
M=10                 # 50 is high end
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # h(x)
hm = (hm + conj(reverse(hm)))/2                     # make h(x) real
x=1.9; @printf "test evalh @ x=%g: " x; println(evalh(hm,x))
x = 2π*rand(1000)
@printf "evalh fast vs slow: %g\n" norm(evalh_slo(hm,x)-evalh(hm,x),Inf)
if dotime>0
    @btime evalh(hm,x)        # why so many allocs & RAM?
    @btime evalh_slo(hm,x)
    t_ns = minimum((@benchmark evalh(hm,x)).times)
    @printf "evalh %g G mode-targs/sec\n" (2M+1)*length(x)/t_ns   # still slow!
    # why allocs?
end

η=1e-6; ω=0.5; tol=1e-8;
@printf "time realadap for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Aa = realadap(hm,ω,η,tol=tol, verb=1)
if dotime>0
    @btime realadap(hm,ω,η,tol=tol)
end
    
#println("roots test, should be +-im: ", roots([1.0,0,1.0]))
# note a can't be a row-vec (matrix)
roots(complex([1.0,0,1.0]))       # complex case
roots([1.0])  # empty list
roots([0.0])  # all C-#s
if dotime>0
    @printf "time roots for M=%d...\n" M
    @btime roots(hm)
end
    
Ac = imshcorr(hm,ω,η,N=30, verb=1)         # a=1 so Davis exp(-30) ~ 1e-13
@printf "test imshcorr: |Ac-Aa| = %g\n" abs(Ac-Aa)
if dotime>0
    @btime imshcorr(hm,ω,η,N=30)
end

# *** need insure linalg eigvals only uses 1 core, or make quadgk multithread !


# ------------ end module tests---------------


# known band struc case
M=1; hm = OffsetVector(complex([1/2,0,1/2]),-M:M)    # h(x)=cos(x)

# *** test eta=0+ analytic form now 
ω=0.7; η=0.0
Ac = imshcorr(hm,ω,η,N=30, verb=1)         # a=1 so Davis exp(-30) ~ 1e-13
Ac = -Ac # *** seems to be sign err? check eta=0 case
Aa = 2π*im/sqrt(1-ω^2)
@printf "η=0+ test: |Ac-anal| = %g\n" abs(Ac-Aa)


# sweep ***
#ωs = 0:0.01:1.7   # energy range

