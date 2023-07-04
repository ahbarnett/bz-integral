using Int1DBZ
using Printf
using OffsetArrays

# just sandbox for now, not using Test
#@testset "evaluators" begin
#    @test 
#end
# ... would need a way to make @test verbose to see results ! :(

M=200         # max mag Fourier freq index (200 to make fevals slow)
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # F-coeffs of h(x)
hm = (hm + conj(reverse(hm)))/2                     # make h(x) real for x Re

nx = 1000
xtest = (1.9, [1.3], 2π*rand(nx)) # 2π*rand(ComplexF64, nx))
# note complex x case fails... fourier_kernel is real-only?
for (t,x) in enumerate(xtest)
    @printf "Scalar evalh variants consistency: test #%d...\n" t
    if t==1
        @printf "\tevalh @ x=%g: " x; println(evalh_ref(hm,x))
    end
    @printf "fourier_kernel chk:          %.3g\n" norm(fourier_kernel(hm,x) - evalh_ref(hm,x),Inf)
end
