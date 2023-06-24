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
