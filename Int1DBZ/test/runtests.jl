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

TIME = TimerOutput()
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
print_timer(TIME, sortby=:firstexec)   # otherwise randomizes order!

@printf "miniquadgk...\n"
f(x) = sin(3*x+1)
F(x) = -cos(3*x+1)/3
Ie = F(1)-F(-1)
r = gkrule()
@printf "\t1-seg G-err %.3g, K err %.3g\n" sum(r.gw.*f.(r.x[2:2:end]))-Ie sum(r.w.*f.(r.x))-Ie
seg = applyrule(f,-1.0,1.0,r)
@printf "\t1-seg (applyrule) I=%.12g E=%.3g\n" seg.I seg.E
a,b = 0.0,6.0
Ie = F(b)-F(a)
I, E, segs, numevals = miniquadgk(f,a,b)
@printf "\tminiquadgk err %.3g, vs reported E %.3g\n" I-Ie E
I, E, segs, numevals = miniquadgk(f,a,b,rtol=1e-12)
@printf "\tminiquadgk err %.3g, vs reported E %.3g\n" I-Ie E

#=
d = 0.1
z0 = 0.3+1im*d
f(x) = sin(x)/(x-z0)
r = gkrule()
=#