# test few_poly_roots. 7/15/23
using Int1DBZ
using Printf
using LinearAlgebra
BLAS.set_num_threads(1)

r = gkrule()       # one seg quadr rule
fac = lu(r.x.^(0:14)')   # has fixes 15 nodes
z0 = 0.3+0.5im
g(x) = sin(pi*(x-z0))     # complex sin, roots @ z0+integers
y = g.(r.x)

# warmup 
roots,derivs = find_near_roots(y, r.x, fac=fac, meth="PR");
println("found # roots: ", length(roots))
show(roots)

# our func (check its debug mode)
c = fac \ y   # coeffs
roots, rvals = few_poly_roots(c, y, r.x, 3, debug=1)
println("r=",roots); println("|rvals|=", abs.(rvals))
println("true func |g(r)|=", abs.(g.(roots)))

#y = ones(ComplexF64,15)   # make value data
#y[2:4:end] .*= -1       # slower osc, half the roots

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1
@btime few_poly_roots($c, $y, $r.x, 1)
@btime few_poly_roots($c, $y, $r.x, 2)
@btime few_poly_roots($c, $y, $r.x, 3)
#@btime find_near_roots($y, $r.x, fac=$fac);

roots,derivs = find_near_roots(y, r.x, fac=fac, meth="F");
println("found # roots: ", length(roots))
show(roots)
@btime find_near_roots($y, $r.x, fac=$fac, meth="F")
