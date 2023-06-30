using Int1DBZ
using OffsetArrays
using Random.Random

#function doit()
Random.seed!(0)
M=10         # max mag Fourier freq index
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # F-coeffs of h(x)
hm = (hm + conj(reverse(hm)))/2                     # make h(x) real for x Re
η=1e-6; ω=0.5; tol=1e-8;
function fwrapper(hm,ω,η;tol=1e-8)
    f(x::Number) = inv(complex(ω,η) - fourier_kernel(hm,x));  # integrand for 1DBZ. Does it matter where defined, global, or in function?
    return miniquadgk(f,0.0,2π,rtol=tol)
end
Am, E, segs, numevals = fwrapper(hm,ω,η,tol=tol)

    
# https://docs.julialang.org/en/v1/manual/profile/
# @profile (for i=1:1000; Am, E, segs, numevals = miniquadgk(f,0.0,2π,rtol=tol); end)
# Profile.print(format=:flat)
# collected around 1900 "backtraces" (samples), claims:
# 83% time on f(x) eval, of which 30% on fourier_kernel.

# See: https://github.com/JuliaPerf/PProf.jl
# using Profile, PProf
# Profile.Allocs.clear()
# Profile.Allocs.@profile (for i=1:1000; Am, E, segs, numevals = miniquadgk(f,0.0,2π,rtol=tol); end)
# PProf.Allocs.pprof()
# ... takes 30 sec for 2GB allocs
# Graph (paste URL to browser) shows 100% Unknown Type.    Not useful!

# SOLUTION: cannot time a function with func handle passed in from outside
# the @time call. Need to wrap the function inside!
