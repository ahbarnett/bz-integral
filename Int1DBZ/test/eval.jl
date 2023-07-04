# test the Fourier series evaluators
using Int1DBZ
using Printf
using OffsetArrays
using StaticArrays
using LinearAlgebra

n=5           # 1 for scalar, else matrix size of H
M=100         # max mag Fourier freq index (200 to make fevals slow)
mlist = -M:M  # OV of SAs version
Hm = OffsetVector([SMatrix{n,n}(randn(ComplexF64,(n,n))) for m in mlist], mlist)
Hmconj = OffsetVector([Hm[m]' for m in mlist], mlist)   # ugh! has to be better!
Hm = (Hm + reverse(Hmconj))/2                           # H(x) hermitian if x Re

nx = 1000
xtest = (1.9, [1.3], 2π*rand(nx)) # 2π*rand(ComplexF64, nx))
# note complex x case fails... fourier_kernel is real-only?
for (t,x) in enumerate(xtest)
    @printf "n=%d (matrix size), evalh variants consistency: test #%d...\n" n t
    if t==1
        @printf "\tevalh @ x=%g: " x; println(evalh_ref(Hm,x))
    end
    @printf "fourier_kernel chk:          %.3g\n" norm(fourier_kernel(Hm,x) - evalh_ref(Hm,x),Inf)
end
