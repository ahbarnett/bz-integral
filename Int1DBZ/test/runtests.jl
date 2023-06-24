using Int1DBZ
#using Test
using Printf
using OffsetArrays
using LinearAlgebra

using TimerOutputs

#@testset "evaluators" begin
#    
#    @test 
#end
# need a way to make @test verbose to see results !

# just sandbox for now

M=10         # max mag Fourier freq index
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

const TIME = TimerOutput()
η=1e-6; ω=0.5; tol=1e-8;
@printf "\nConventional quadrature (eta>0, obvi):\n"
@printf "test realadap for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Aa = realadap(hm,ω,η,tol=tol, verb=1)
@printf "\tAa = "; println(Aa)
TIME(realadap)(hm,ω,η,tol=tol, verb=1)
@printf "test realadap_lxvm for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Al = realadap_lxvm(hm,ω,η,tol=tol, verb=1)
@printf "\tAl = "; println(Aa)
TIME(realadap_lxvm)(hm,ω,η,tol=tol, verb=1)

show(TIME, sortby=:firstexec)   # use sortby otherwise randomizes order!
