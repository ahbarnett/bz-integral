# Generalized Chebychev quadrature tool.
# Barnett 1/13/24

using Int1DBZ
using FastGaussQuadrature
using LinearAlgebra

"""
   x, w = genchebquad(fs, a, b, tol) returns quadrature nodes `x`` in [a,b] and
   corresponding (possibly complex-values) weights `w`` that integrate all
   (real or complex valued) functions defined by fs on (a,b) to
   relative tolerance `tol`. This is done via generalized Chebychev
   quadrature. fs must be a single vector-valued function with scalar argument.
   *** add refs.

   Notes: uses miniquadgk on fs to choose the initial dense node set, so can
   get its used segments out. 
"""
function genchebquad(fs::Function, a, b, tol=1e-10)
    I,E,segs,numeval = miniquadgk(fs,Float64(a),Float64(b); rtol=tol)
    Nf = length(I)     # how many funcs
    z0,w0 = gausslegendre(23)       # since 2*23-1=45 > twice GK(7,15) order of 22
    # doubling of order allows integrating all products to tol
    x = [(s.a+s.b)/2 .+ (s.b-s.a)/2*z0 for s in segs]          # dense nodes
    x = reduce(vcat, x)               # flatten to one vector
    w = [(s.b-s.a)/2*w0 for s in segs]          # dense weights
    w = reduce(vcat, w)               # flatten to one vector
    println("nfuncs=",Nf,"\ttol=",tol,"\tnsegs=",length(segs),
        "\t m(#densenodes)=",length(x))
     # expensive fill A size m*Nf...
    A = reduce(vcat, [sqrt(wj)*fs(xj) for (xj,wj) in zip(x,w)]')
    S = svd(A)    # reduced
    r = sum(S.S .> tol*S.S[1])
    println("rank=",r)
    U = S.U[:,1:r]
    F = qr(U',ColumnNorm())            # CPQR to get nodes in 1...m
    nodeinds = F.p[1:r]                # permutation vector
    x = x[nodeinds]                    # final subset of nodes
    # Vandermonde: vals of U-funcs at these nodes...
    V = Diagonal(1.0./sqrt.(w[nodeinds]))*U[nodeinds,:] 
    Is = transpose(sum(Diagonal(sqrt.(w))*U, dims=1))   # col vec, u func integrals
    w = transpose(V) \ Is              # solve trans Vandermonde, match u integrals
    println("transp-Vander resid=",norm(transpose(V)*w - Is))
    x, w[:]                            # make w col vec too
end

# test codes
fs(x::Number) = [x^k for k=0:40]        # monomials
x, w = genchebquad(fs,-1,1,1e-10)
using GLMakie
fig,ax,l = scatter(x,w); linesegments!(kron(x,[1;1]),kron(w,[0;1]))
display(fig)

