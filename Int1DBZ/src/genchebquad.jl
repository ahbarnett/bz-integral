# Generalized Chebychev quadrature tool.
# Barnett 1/13/24

using Int1DBZ
using FastGaussQuadrature
using LinearAlgebra
using Printf

struct gcq_info
    segs
    xg
    wg
    U
end

"""
    x, w = genchebquad(fs, a, b[, tol]; verb=0)
    
Use generalized Chebychev quadrature (GCQ) to compute a vector of quadrature
nodes `x`, each nodes in (a,b), and a corresponding vector of weights `w`, that
integrate all functions defined by `fs` on (a,b), to relative tolerance `tol`.
`fs` should be a vector-valued function of a scalar argument, where each entry
of the vector defines a function in the set. `verb>0` gives text diagnostics.

# Example
```julia-repl
fs(x) = [x^r for r=(-1:30)/2]          # set of powers -1/2,0,1/2,1,... 
x,w = genchebquad(fs, 0,1, 1e-12)      # returns 21 nodes and weights
```

Notes:
1) If `fs` has complex outputs, the weights `w` will also be complex. They work,
   but are not very convenient. Real-valued `fs` is recommended (eg, send in Re,
   Im parts as separate functions).
2) It uses miniquadgk on `fs` to choose the initial dense node set, so can get
   its used segments out. 

References:
* Sec. 4.3 of: J. Bremer, Z. Gimbutas, and V. Rokhlin, "A nonlinear optimization
     procedure for generalized Gaussian quadratures," SIAM J. Sci. Comput.
        32(4), 1761--1788 (2010).
* App. B of: D. Malhotra and A. H. Barnett, https://arxiv.org/abs/2310.00889
"""
function genchebquad(fs::Function, a, b, tol=1e-10; verb=0)
    # Version also returning diagnostic outputs. verbosity verb=0,1,...
    I,E,segs,numeval = miniquadgk(fs,Float64(a),Float64(b); atol=tol, rtol=tol)
    sort!(segs; lt = (s,t) -> s.a<t.a)        # reorder segs along real axis
    Nf = length(I)     # how many funcs
    z0,w0 = gausslegendre(14)       # gets twice GK(7,15) min order of 2*7-1=13
    # doubling of order allows integrating all products to tol
    xg = [(s.a+s.b)/2 .+ (s.b-s.a)/2*z0 for s in segs]          # dense nodes
    xg = reduce(vcat, xg)               # flatten to one vector
    wg = [(s.b-s.a)/2*w0 for s in segs]          # dense weights (real >0)
    wg = reduce(vcat, wg)               # flatten to one vector
    if verb>0; println("nfuncs=",Nf,"\ttol=",tol,"\tnsegs=",length(segs),
        "\t m(#densenodes)=",length(xg)); end
     # expensive fill A size m*Nf...
    A = reduce(vcat, [sqrt(wj)*fs(xj) for (xj,wj) in zip(xg,wg)]')
    S = svd(A)    # reduced
    r = sum(S.S .> tol*S.S[1])         # *** allow setting rank instead of tol
    if verb>0; println("rank=$r"); end   #"\tU_11=",S.U[1,1])
    U = S.U[:,1:r]
    F = qr(U',ColumnNorm())            # CPQR to get nodes in 1...m
    nodeinds = F.p[1:r]                # permutation vector
    x = xg[nodeinds];                  # final subset of nodes
    # Vandermonde: vals of U-funcs at these nodes...
    V = Diagonal(1.0./sqrt.(wg[nodeinds]))*U[nodeinds,:] 
    Is = transpose(sum(Diagonal(sqrt.(wg))*U, dims=1))  # col vec of u func ints
    w = V' \ Is              # solve trans Vandermonde sys to match u integrals
    # *** fs complex not recommended; here for V complex why not transp?
    if verb>0; @printf "transp-Vander resid=%.3g\n" norm(V'*w - Is); end
    info = gcq_info(segs,xg,wg,U)   # save diagnostics
    x, w[:], info      # make w col vec too. not yet sorted wrt x_j
end
