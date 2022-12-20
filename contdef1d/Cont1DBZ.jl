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
    roots

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



end

