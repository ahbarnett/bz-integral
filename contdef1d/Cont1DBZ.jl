module Cont1DBZ
"""
Cont1DBZ: module for 1D Brillouin zone integration via contour deformation.

A H Barnett, Dec 2022
"""

using OffsetArrays
using QuadGK
using LinearAlgebra
using Printf

export
    evalh,
    evalh_slo,
    evalhp,
    realadap,
    roots,
    imshcorr

"""
    evalh(hm,x)

    evaluate band Hamiltonian h(x) as complex Fourier series with coeffs hm
    (an offsetvector), at x, a target or vector of targets (real or complex)
    Use phase-winding method.
"""
function evalh(hm,x)
    N = length(x)
    Tout = eltype(complex(hm))
    h = zeros(Tout,size(x))
    dph = exp.(im*x)            # phase to wind by. (don't assume x real)
    mmin = 1+hm.offsets[1]       # get start & stop freq indices
    mmax = mmin+length(hm)-1
    ph = exp.(im*mmin*x)
    for m=mmin:mmax
        h = h .+ hm[m]*ph
        ph = ph.*dph
    end
    h
end

"""
    evalh_slo(hm,x) - slow version of evalh
"""
function evalh_slo(hm,x)
    N = length(x)
    Tout = eltype(complex(hm))
    h = zeros(Tout,size(x))
    for m in eachindex(hm)
        h = h .+ hm[m]*exp.(im*m*x)
    end
    h
end

"""
    evalhp(hm,x)

    evaluate band Hamiltonian derivative h'(x) as complex Fourier series with
    coeffs hm (an offsetvector), at x, a target or vector of targets.
    Slow version.
"""
function evalhp(hm,x)
    N = length(x)
    Tout = eltype(complex(hm))
    hp = zeros(Tout,size(x))
    for m in eachindex(hm)
        hp = hp .+ hm[m]*im*m*exp.(im*m*x)
    end
    hp
end

"""
    A = realadap(hm,ω,η;tol,verb)

    use quadgk on Re axis to integrate 1/(h(x) - ω + iη).
    hm is given by offsetvector of Fourier series. tol controls rtol.
"""
function realadap(hm,ω,η; tol=1e-8, verb=0)
    f(x) = 1 ./ (evalh(hm,x) .+ (-ω+im*η))     # integrand func (x can be vec)
    A,err = quadgk(f,0,2π,rtol=tol)          # can't get more info? # fevals?
    if verb>0
        @printf "\trealadap err=%g\n" err
    end
    A
end

"""
    roots(a)

    find all complex roots of polynomial a[1]*z^n + a[2]*z^(n-1) + ... + a[n+1]
    via companion matrix EVP in O(n^3) time. Similar to MATLAB roots.
"""
function roots(a::Vector)
    a = vec(a)              # make sure not funny shaped matrix
    a = complex(a)          # idempotent, unlike Complex{T} for T a type...
    T = eltype(a)
    while length(a)>1 && a[1]==0.0         # gobble up any zero leading coeffs
        a = a[2:end]
    end
    if isempty(a) || (a==[0.0])            # done, meaningless
        return NaN
    end
    deg = length(a)-1       # a is now length>1 with nonzero 1st entry
    if deg==0
        return T[]          # done: empty list of C-#s
    end
    a = reshape(a[deg+1:-1:2] ./ a[1],(deg,1))    # make monic, col and flip
    C = [ [zeros(T,1,deg-1); Matrix{T}(I,deg-1,deg-1)] -a ]   # stack companion mat
    eigvals!(C)              # overwrite C, and we don't want the vectors
end
# Note re don't need evecs: see also LinearAlgebra.LAPACK.geev!
roots(a::OffsetVector) = roots(a.parent)   # handle OV's


"""
    A = imshcorr(hm,ω,η;N,s,a,verb)

    integrate 1/(h(x) - ω + iη) from 0 to 2π using imag shift PTR contour
    plus residue theorem corrections and pole-subtraction by cotangents.
    hm is given by offsetvector of Fourier series indices -M:M.
    η may be >0 or =0 (giving lim -> 0+).
"""
function imshcorr(hm,ω,η; N::Int=20, s=1.0, a=1.0, verb=0)
    hmplusc = copy(hm);                # otherwise changing hmconst changes hm!
    hmplusc[0] = hm[0] + (-ω+im*η)     # F series for denominator
    M = -hm.offsets[1]-1
    zr = roots(reverse(hmplusc))       # shift powers by M, use poly coeff ord
    xr = @. log(zr)/im                 # solve z = e^{ix} for roots x of denom
    xr = @. mod(real(xr),2π) + im*imag(xr)    # fold Re to [0,2π), for humans
    hpxr = evalhp(hm,xr)               # h' at all roots *** maybe a waste?
    
    eps = 1e-13    # dist from Re axis considered real when η=0
    # *** todo choose N,s,a via M, xr (eg s staying >1e-3 from all imag(xr))
    if verb>0
        @printf "imshcorr: N=%d s=%g a=%g eps=%g...\n" N s a eps
    end
    xj = im*s .+ 2π*(1:N)/N            # PTR nodes on imag-shifted contour
    w = 2π/N                           # all weights
    fj = 1.0 ./ evalh(hmplusc, xj)     # bare integrand at nodes
    A = w*sum(fj)                      # bare PTR
    for r in eachindex(xr)             # all poles (roots of denom)
        imx = imag(xr[r])
        resthm = (0.0 < imx < s)
        if η==0.0 && abs(imx)<eps      # update whether to apply residue thm
            resthm = (real(hpxr[r]) < 0.0)     # so eta->0+ gives Re pole ->0+
        end
        res = 1.0 ./ hpxr[r]           # Res of f = 1/h'(x_r)
        A = A + 2π*im * res * resthm
        cotcorr = (abs(imx-s) < a)
        if cotcorr
            cotj = @. cot((xj-xr[r])/2)        # cot at nodes
            # *** speed up cot eval via phase-stepping for cos, sin?
            A = A - (w*res/2)*sum(cotj)        # pole-cancel correct to PTR
            cotsign = (2*(imx>s)-1)    # sign of analytic cot integral
            A = A + cotsign * π*im * res
        end
        if verb>0
            @printf "\tpole %g+%gi:   \t resthm=%d \t cotcorr=%d\n" real(xr[r]) imx resthm cotcorr
        end
    end
    A
end

end
