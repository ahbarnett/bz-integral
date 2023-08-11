# tester for global root finding, scalar and matrix. Barnett 7/4/23
using Int1DBZ
using Printf
using OffsetArrays
using StaticArrays
using LinearAlgebra
using Random.Random

for n = 1:2             # matrix size for H (1=scalar).. Check scalar & matrix
    M = 10            # max mag Fourier freq index (eg 200 to make fevals slow)
    η = 1e-5
    ω = 0.5
    Random.seed!(0)         # set up 1D BZ h(x) for denominator
    if n == 1           # scalar case without SMatrix-valued coeffs
        Hm = OffsetVector(randn(ComplexF64, 2M + 1), -M:M)  # F-coeffs of h(x)
        Hm = (Hm + conj(reverse(Hm))) / 2                 # make h(x) real for x Re
    else
        mlist = -M:M    # OV of SAs version
        Hm = OffsetVector([SMatrix{n,n}(randn(ComplexF64, (n, n))) for m in mlist], mlist)
        Hmconj = OffsetVector([Hm[m]' for m in mlist], mlist) # ugh! has to be better!
        Hm = (Hm + reverse(Hmconj)) / 2                     # H(x) hermitian if x Re
    end

    # test global root-finding on integrand
    xr = BZ_denominator_roots(Hm, ω, η)
    @printf "found %d roots...\n" length(xr)
    for x in xr
        if n == 1
            @printf "denom mag @ x = %g + %gi, should vanish: %.3g\n" real(x) imag(x) abs(complex(ω, η) - fourier_kernel(Hm, x))
        else    # note below I was dangerous since variable overrides Id operator!
            U, S, V = svd(complex(ω, η)*LinearAlgebra.I - fourier_kernel(Hm, x))
            @printf "min sing val @ x=%g+%gi should vanish %.3g\n" real(x) imag(x) minimum(S)
        end
    end

end
