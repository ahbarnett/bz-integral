module Cont1DBZ
"""
Cont1DBZ: module for 1D Brillouin zone integration via contour deformation.

A H Barnett, help by LXVM. Dec 2022 - Jan 2023.
"""

using OffsetArrays
using QuadGK
using LinearAlgebra
using Printf
using AMRVW               # roots in O(N^2)
using PolynomialRoots     # low-order faster roots

using LoopVectorization    # experimental

export
    evalh_ref,
    evalh_wind,
    evalh,
    evalhp,
    realadap,
    roots,
    roots_best,
    imshcorr,
    discresi,
    fourier_kernel,
    realadap_lxvm


"""
    ph_type(x)

Helper function that returns the type of output needed to store the values of
Fourier coefficients (or 'ph'ase)

By LXVM
"""
ph_type(x) = Base.promote_op(cis, eltype(x))

"""
    hx_type(hm, x)

Helper function that returns the type of output needed to store the result of a
Fourier series, eg h(x)

By LXVM
"""
hx_type(hm,x) = Base.promote_op(*, eltype(hm), ph_type(x))

"""
    evalh_ref(hm,x) - slow version of evalh; reference implementation

    evaluates band Hamiltonian h(x) as complex Fourier series with coeffs hm
    (an offsetvector), at x, a target or vector of targets (real or complex)
"""
function evalh_ref(hm,x::Number)
    h = zero(hx_type(hm,x))                  # preserves type
    for m in eachindex(hm)
        h += hm[m]*exp(im*m*x)
    end
    h
end
evalh_ref(hm,x::AbstractArray) = map(y -> evalh_ref(hm,y), x)

"""
    evalh_wind(hm,x)

    evaluate band Hamiltonian h(x) as complex Fourier series with coeffs hm
    (an offsetvector), at x, a target or vector of targets (real or complex)
    Use phase-winding method.

    Dev notes: Turns out writing out the i loop is faster than vector notation
"""
function evalh_wind(hm,x::AbstractArray)
    # allocate arrays
    h = zeros(hx_type(hm,x), size(x))
    ph = zeros(ph_type(x), size(x)); dph = similar(ph)
    # allocation-free kernel
    @. dph = cis(x)            # phase to wind by. (don't assume x real)
    mmin = 1+hm.offsets[1]       # get start & stop freq indices
    mmax = mmin+length(hm)-1
    @. ph = cis(mmin*x)
    for m=mmin:mmax             # this loop must be sequential
        @inbounds @fastmath for i in eachindex(h)   # this loop triv parallelizable, but complex arith so no @avx
            h[i] += hm[m]*ph[i]
            ph[i] *= dph[i]
        end
    end
    h
end
function evalh_wind(hm,x::Number) # scalar version, nonallocating
    mmin = 1+hm.offsets[1]       # get start & stop freq indices
    mmax = mmin+length(hm)-1
    h = zero(hx_type(hm,x))
    ph = exp(im*mmin*x)
    dph = exp(im*x)
    @inbounds @fastmath for m=mmin:mmax
        h += hm[m] * ph
        ph *= dph
    end
    h
end

"""
    evalh(hm,x)

    evaluate band Hamiltonian h(x) as complex Fourier series with coeffs hm
    (an offsetvector), at x, a target or vector of targets (real or complex).
    Use phase-winding method & LoopVectorization / multithreading (?)
    @avx is 50% faster than @axvt (multithr).
"""
function evalh(hm::AbstractVector{<:Number},x::AbstractArray)
    # lxvm's low-allocation set-up...
    # allocate arrays
    hc = zeros(hx_type(hm, x), size(x))
    hr = real(hc)
    hi = imag(hc)
    dphr = Vector{real(ph_type(x))}(undef, size(x))
    dphi = Vector{real(ph_type(x))}(undef, size(x))
    phr  = Vector{real(ph_type(x))}(undef, size(x))
    phi  = Vector{real(ph_type(x))}(undef, size(x))
    # allocation-free kernel
    mmin = 1+hm.offsets[1]       # get start & stop freq indices
    mmax = mmin+length(hm)-1
    for (i, e) in enumerate(x)
        phr[i], phi[i] = reim(cis(mmin*e))          # starting phase (don't assume x real)
        dphr[i], dphi[i] = reim(cis(e))             # phase to wind by (don't assume x real)
    end
    for m=mmin:mmax             # this loop must be sequential
        hmr, hmi = reim(hm[m])
        @avx for i in eachindex(phr)   # this loop triv par (& avx-ble since Re)
            hr[i] += hmr*phr[i] - hmi*phi[i]  # complex arith via reals
            hi[i] += hmi*phr[i] + hmr*phi[i]  # NB if hi scalar, setindex! borks
            tr = dphr[i]*phr[i] - dphi[i]*phi[i]   # temp vars for clean update
            ti = dphi[i]*phr[i] + dphr[i]*phi[i]
            phr[i] = tr
            phi[i] = ti
        end
    end
    @. hc = complex(hr, hi)
    hc
end
# avx acceleration fails for matrix coefficients, so defining fallbacks below
evalh(hm,x::Number) = evalh_wind(hm,x)
evalh(hm::AbstractVector{<:AbstractMatrix},x::AbstractArray) = map(y -> evalh(hm,y), x)

"""
    evalhp(hm,x)

    evaluate band Hamiltonian derivative h'(x) as complex Fourier series with
    coeffs hm (an offsetvector), at x, a target or vector of targets.
    Slow reference version.
"""
function evalhp(hm,x::Number)
    hp = zero(hx_type(hm,x))                  # preserves type
    for m in eachindex(hm)
        hp += hm[m]*im*m*exp(im*m*x)
    end
    hp
end
evalhp(hm,x::AbstractArray) = map(y -> evalhp(hm,y), x)

"""
    fourier_kernel(C::OffsetVector, x)
    fourier_kernel(C::Vector, x, [myinv=inv])

A non-allocating 1D Fourier series evaluator that assumes the input Fourier
coefficients `C` are an `OffsetVector` with symmetric indices (i.e. `-m:m`). The
optional argument `myinv` is specialized to `conj` when `x isa Real` since that
is when the twiddle factors are roots of unity.

By LXVM
"""
@inline fourier_kernel(C::OffsetVector, x) = fourier_kernel(C.parent, x)
fourier_kernel(C::Vector, x::Real) = fourier_kernel(C, x, conj) # z = cis(x) is a root of unit so inv(z) = conj(z)
fourier_kernel(C::OffsetVector, x::AbstractArray) = map(y -> fourier_kernel(C,y), x)   # handle arrays
function fourier_kernel(C::Vector, x, myinv=inv)
    s = size(C,1)
    isodd(s) || return error("expected an array with an odd number of coefficients")
    m = div(s,2)
    @inbounds r = C[m+1]
    z₀ = cis(x)
    z = one(z₀)
    @fastmath @inbounds for n in Base.OneTo(m)
        z *= z₀
        r += z*C[n+m+1] + myinv(z)*C[-n+m+1] # maybe this loop layout invites cache misses since the indices are not adjacent?
    end
    r
end


####### Conventional real-axis quadrature methods...

"""
    A = realadap(hm,ω,η;tol,verb)

    use quadgk on Re axis to integrate 1/(ω - h(x) + iη).
    hm is given by offsetvector of Fourier series. tol controls rtol.
"""
function realadap(hm,ω,η; tol=1e-8, verb=0, kernel=evalh)
    f(x::Number) = tr(inv(complex(ω,η)*I - kernel(hm,x)))    # integrand func (quadgk gives x a number)
    A,err = quadgk(f,0,2π,rtol=tol)          # can't get more info? # fevals?
    if verb>0
        @printf "\trealadap claimed err=%g\n" err
    end
    A
end

"""
    A = realadap_lxvm(hm,ω,η;tol,verb)

    use quadgk on Re axis to integrate 1/(ω - h(x) + iη), via
    faster non-allocating 1D Fourier series evaluator.
    hm is given by offsetvector of Fourier series. tol controls rtol.
    By LXVM.
"""
realadap_lxvm(hm, ω, η; tol=1e-8, verb=0) = realadap(hm, ω, η; tol=tol, verb=verb, kernel=fourier_kernel)



######## Methods relating to new contour-deformation techniques...

"""
    roots(a)

    find all complex roots of polynomial a[1]*z^n + a[2]*z^(n-1) + ... + a[n+1]
    via companion matrix EVP in O(n^3) time. Similar to MATLAB roots.
    Note poly coeffs are in reverse order that in many Julia pkgs.
    If the entire C plane is a root, returns [complex(NaN)].
"""
function roots(a::AbstractVector)    # does not allow dims>1 arrays
    a = complex(a)          # idempotent, unlike Complex{T} for T a type...
    T = eltype(a)
    while length(a)>1 && a[1]==0.0         # gobble up any zero leading coeffs
        a = a[2:end]
    end
    if isempty(a) || (a==[0.0])            # done, meaningless
        return [complex(NaN)]     # array, for type stability. signifies all C
    end
    deg = length(a)-1       # a is now length>1 with nonzero 1st entry
    if deg==0
        return T[]          # done: empty list of C-#s
    end
    a = reshape(a[deg+1:-1:2] ./ a[1],(deg,1))    # make monic, col and flip
    C = [ [zeros(T,1,deg-1); Matrix{T}(I,deg-1,deg-1)] -a ]   # stack companion mat
    # at this point we want case of real C to be possible, faster than complex
    complex(eigvals!(C))    # overwrite C, and we don't want the vectors
end
# Note re don't need evecs: see also LinearAlgebra.LAPACK.geev!

"""
    roots_best(a)

    find all complex roots of polynomial a[1]*z^n + a[2]*z^(n-1) + ... + a[n+1]
    using a wrapper to a (supposedly) optimal choice of method.
    Same interface as MATLAB roots, and roots().
"""
function roots_best(a::AbstractVector)    # does not allow dims>1 arrays
    if length(a)<220              # we're pushing it, may die in acc for >200
        PolynomialRoots.roots(reverse(a))
    else              # stable O(M^2) but not quite as fast
        AMRVW.roots(reverse(a))
    end
end


######## The main new methods...

"""
    A = imshcorr(hm,ω,η;N,s,a,verb)

    integrate 1/(ω - h(x) + iη) from 0 to 2π using imag shift PTR contour
    plus residue theorem corrections and pole-subtraction by cotangents.
    hm is given by OffsetVector of Fourier series indices -M:M.
    η may be >0 or =0 (giving lim -> 0+).
    h(x) is scalar (n=1) only for now.
"""
function imshcorr(hm::AbstractVector{<:Number},ω,η; N::Int=20, s=1.0, a=1.0, verb=0)
    hmplusc = -hm;                     # was copy(hm) else hmconst changes hm!
    hmplusc[0] += ω+im*η               # F series for denominator
    #M = -hm.offsets[1]-1              # not yet needed
    hmplusc_vec = hmplusc.parent       # shift powers by M: data vec inds 1:2M+1
    zr = roots_best(reverse(hmplusc_vec))   # flip to use poly coeff ord
    xr = @. log(zr)/im                 # solve z = e^{ix} for roots x of denom
    xr = @. mod(real(xr),2π) + im*imag(xr)    # fold Re to [0,2π), for humans
    dpxr = evalhp(hmplusc,xr)          # denom' (= -h') at all roots
    
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
            resthm = (real(dpxr[r]) > 0.0)     # so eta->0+ gives Re pole ->0+
        end
        res = 1.0 ./ dpxr[r]           # Res of f = 1/denom'(x_r)
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

"""
    A = discresi(hm,ω,η;verb)

    integrate 1/(ω - h(x) + iη) from 0 to 2π using residue theorem for the
    disc |z|<1 where z = exp(ix).
    hm is given by OffsetVector of Fourier series coeffs with indices -M:M.
    η may be >0 or =0 (giving lim -> 0+).
    h(x) is scalar (n=1) only for now.
"""
function discresi(hm::AbstractVector{<:Number},ω,η; verb=0)   # only for n=1
    hmplusc = -hm;                     # was copy(hm) else hmconst changes hm!
    hmplusc[0] += ω+im*η               # F series for denominator
    hmplusc_vec = hmplusc.parent       # shift powers by M: data vec inds 1:2M+1
    zr = roots_best(reverse(hmplusc_vec))   # flip to use poly coeff ord
    A = complex(0.0)                   # CF64
    for z in zr                        # all z poles (roots of denom)
        if abs(z)<1.0     # *** need add |z| approx 1 case via deriv chk ***
            res = -1.0/evalhp(hm,log(z)/im)    # send in x corresp to z
            A += 2π*im*res
            verb==0 || @printf "\tpole |z|=%.14f ang=%.6f: \tres=%g+%gi\n" abs(z) angle(z) real(res) imag(res)
        elseif verb>0
            @printf "\tpole |z|=%.15f ang=%.6f: \tignored\n" abs(z) angle(z)
        end
    end
    A
end

end
