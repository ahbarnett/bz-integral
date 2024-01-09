# see if can use root of poly-fit on 1 segment nodes to find inv-sqrt sing.
# Barnett, 10/3/23.

using Int1DBZ
using GLMakie
verb=1

# single [-1,1] interval
d = 1e-3         # imag dist
z0 = 0.3+1im*d    # sing loc
f(x::Number) = 1.0 + sqrt(-1im/sin(x-z0))     # sqrt sing @ z0, branch cut not hit interval. Next root dist ~1.8 from [a,b]
#f(x::Number) = sqrt(-1im/(x-z0))     # plain sqrt sing @ z0
# key: 1/sqrt(anal) does not capture all the possible
# of form anal(z) + anal(z)/sqrt(z-z0)

r = gkrule()     # GK rule on [-1,1]
xj = r.x    # the larger node set
fj = f.(xj)
if verb>0
    t = range(-1.0,1.0,1000)          # for cont curve plot
    fig,ax,=scatter(xj,real.(fj), label=L"Re $f_j$")       # nodes
    scatter!(xj,imag.(fj), label=L"Im $f_j$")
    lines!(t,real.(f.(t)))           # curve
    lines!(t,imag.(f.(t)))
    ax.xlabel=L"$t$"
    axislegend()
end

Ig, Eg, segs, numevals = miniquadgk(f,-1.0,1.0,rtol=1e-12);   # "exact"
gj = 1.0./fj.^2   # g will have root where f has pow -1/2 sing.

rho = exp(1.0)       # Bernstein ellipse to accept out to
roots, gproots = find_near_roots(gj, xj, rho=rho, meth="PR")  # steps 1+2
println("err in locating sing = ",abs(roots[1]-z0))
# concl: we can reliably fit the (-0.5)-power sing loc, only if f has a *factor* sqrt, but not when general

Ising = 0.0        # add up analytic contribs?
for (i,r) in enumerate(roots)
    Res = 1.0/gproots[i]    # step 3
    println("\tpole #",i,": r=",r,",  1/|f(r)|^2=",1.0/abs(f(r))^2,"  Res=",Res)
    @. fj -= Res/(xj-r)       # subtract, step 4
    global Ising +=  Res * log((1.0-r)/(-1.0-r))
end
