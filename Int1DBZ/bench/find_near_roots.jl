# is find_near_roots slowed by allocations or horner?
using Int1DBZ
using Profile
using LinearAlgebra

r = gkrule()
fac = lu(r.x.^(0:14)')
y = ones(ComplexF64,15)   # make value data
#y = rand(ComplexF64,15)   # make value data
#y[2:2:end] .*= -1       # alternating vals, lots of roots in [-1,1]
y[2:4:end] .*= -1       # slower osc, half the roots
#y[3:4:end] .*= -1       # "
#y = complex(r.x)      # 1 root
#y = complex(sin.(5*r.x))  # 5 roots in rho

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1
roots,derivs = find_near_roots(y, r.x, fac=fac); println(length(roots)) # warmup
#@btime find_near_roots($y, $r.x);
@btime find_near_roots($y, $r.x, fac=$fac);

# with fac: 1.6us for solve c.
# roots: varies from 0us (no roots) thru 10us (typ) thru 22us (a lin func?!)
# [finding all 14 derivs used to add  64us :( due to poor use of horner :(]
# now derivs adds ~1us.
