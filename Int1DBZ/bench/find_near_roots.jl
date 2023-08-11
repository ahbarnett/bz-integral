# is find_near_roots slowed by allocations or horner?
using Int1DBZ
using Profile
using LinearAlgebra
BLAS.set_num_threads(1)

r = gkrule()
fac = lu(r.x.^(0:14)')
y = ones(ComplexF64,15)   # make value data
#y = rand(ComplexF64,15)   # make value data
#y[2:2:end] .*= -1       # alternating vals, lots of roots in [-1,1]
#y[2:4:end] .*= -1       # slower osc, half the roots
#y[3:4:end] .*= -1       # "
#y = complex(r.x)      # 1 root
#y = complex(sin.(5*r.x))  # 5 roots in rho
y = complex(sin.(π*(r.x.-0.3)))  # 3 roots at -0.7, +0.3, +1.3

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1

roots,derivs = find_near_roots(y, r.x, fac=fac);    # warmup
println("found # roots: ", length(roots))
@btime find_near_roots($y, $r.x);
@btime find_near_roots($y, $r.x, fac=$fac);

# with fac: 1.6us for solve c.  No worse for 1thread than 8threads.
# roots: varies from 0us (no roots) thru 10us (typ) thru 22us (a lin func?!)
# [finding all 14 derivs used to add  64us :( due to poor use of horner :(]
# now derivs adds ~1us.

# This is my own Newton+deflation, now 2.6 us: (up to 3 near roots)
@btime find_near_roots($y, $r.x, fac=$fac, meth="F");

if false       # try flamegraph. Nothing like as useful as vscode flamegraph
    BenchmarkTools.DEFAULT_PARAMETERS.seconds=1.0           # get to 1e4 samps
    @bprofile find_near_roots($y, $r.x, fac=$fac, meth="F");
    using ProfileView    # annoying long precompile; also, not in Project.toml
    ProfileView.view()        # not so useful -> how see lines in code?
    # can tell Newton dominates deflation, eg.
    using ProfileSVG
    ProfileSVG.save("find_near_roots_flamegraph.svg")
    # almost useless since text overlaps :(
end

if false
    # from vscode:   need to run a sizeable job (few secs)...
    @profview for i=1:1000000; find_near_roots(y, r.x, fac=fac, meth="F"); end
    # but how combine with @bprofile?  dunno

    # view allocs (can't run very long & what's sample_rate??)
    @profview_allocs for i=1:100; find_near_roots(y, r.x, fac=fac, meth="F"); end sample_rate=1
end
