# developing miniquadgk (late June 2023)

using Int1DBZ
using Printf

@printf "miniquadgk on plain smooth func...\n"
f(x::Number) = sin(3*x+1)
F(x::Number) = -cos(3*x+1)/3
Ie = F(1)-F(-1)
r = gkrule()
# note this writing of sum(x.*y) in Julia bad for performance code (allocs)...
@printf "\t1-seg G-err %.3g, K err %.3g\n" sum(r.gw.*f.(r.x[2:2:end]))-Ie sum(r.w.*f.(r.x))-Ie
seg = applygkrule(f,-1.0,1.0,r)
@printf "\t1-seg (applygkrule) I=%.12g E=%.3g\n" seg.I seg.E
a,b = 0.0,6.0
Ie = F(b)-F(a)
I, E, segs, numevals = miniquadgk(f,a,b)
@printf "\tminiquadgk(%d fevals) err %.3g, \testim E %.3g\n" numevals I-Ie E
I, E, segs, numevals = miniquadgk(f,a,b,rtol=1e-12)
@printf "\tminiquadgk(%d fevals) err %.3g, \testim E %.3g\n" numevals I-Ie E

plot(segs,:miniquadgk)   # labeled plot

@printf "perf test vs quadgk in context of real-axis quad of 1/(c+h(x))...\n"

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1
using Random.Random

M=20         # max mag Fourier freq index (200 to make fevals slow)
η=1e-6; ω=0.5; tol=1e-8;  # 1e-8 too much for M=200 realadap to handle :(
Random.seed!(0)
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # F-coeffs of h(x)
hm = (hm + conj(reverse(hm)))/2                     # make h(x) real for x Re

@btime realmyadap(hm,ω,η,tol=tol);
# Before allocated fvals in type-stable way: 842.324 μs (14354 allocations: 515.05 KiB)
# After:  552.549 μs (17 allocations: 45.52 KiB)      close enough for jazz
@btime realadap_lxvm(hm,ω,η,tol=tol);
#  551.587 μs (8 allocations: 36.81 KiB)

# developing realmyadap... which uses miniquadgk on shifted Fourier series...
@printf "test realadap_lxvm for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Al = realadap_lxvm(hm,ω,η,tol=tol, verb=1)
@printf "\tAl = "; println(Al)
f(x::Number) = inv(complex(ω,η) - fourier_kernel(hm,x));  # integrand for 1DBZ
@printf "test miniquadgk on global f for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Am, E, segs, numevals = miniquadgk(f,0.0,2π,rtol=tol)    # NOT for timing...
# ... since f is not "interpolated". Would need to define f in a function
@printf "\tAm = "; println(Am)
@printf "\t\tabs(Am-Al)=%.3g\n" abs(Am-Al)
Am, E, segs, numevals = realmyadap(hm,ω,η,tol=tol)
@printf "test realmyadap (same pars): fevals=%d, nsegs=%d, claimed err=%g\n" numevals length(segs) E
@printf "\t\tabs(Am-Al)=%.3g\n" abs(Am-Al)
