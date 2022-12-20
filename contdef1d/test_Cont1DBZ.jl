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

# -------- module method tests ----------
# test eval h...
M=100
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # h(x)
x=1.9; @printf "test evalh @ x=%g: " x; println(evalh(hm,x))
x = 2π*rand(1000)
@printf "evalh fast vs slow: %g\n" norm(evalh_slo(hm,x)-evalh(hm,x),Inf)
@btime evalh(hm,x)        # why so many allocs & RAM?
@btime evalh_slo(hm,x)
t_ns = minimum((@benchmark evalh(hm,x)).times)
@printf "evalh %g G mode-targs/sec\n" (2M+1)*length(x)/t_ns   # still slow!
# why allocs?

η=1e-6; tol=1e-8; @printf "time realadap for M=%d, η=%g, tol=%g... " M η tol
@btime realadap(hm,0.5,η,tol=tol)

#println("roots test, should be +-im: ", roots([1.0,0,1.0]))
# note a can't be a row-vec (matrix)
roots(complex([1.0,0,1.0]))       # complex case
roots([1.0])  # empty list
roots([0.0])  # all C-#s

# ------------ end module tests---------------


# known band struc case
M=1; hm = OffsetVector(complex([1/2,0,1/2]),-M:M)    # h(x)=cos(x)

ωs = 0:0.01:1.7   # energy range

