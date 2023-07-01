using Int1DBZ
using Printf
using OffsetArrays
using LinearAlgebra
using Random.Random
using TimerOutputs
using Gnuplot

# just sandbox for now, not using Test
#@testset "evaluators" begin
#    @test 
#end
# ... would need a way to make @test verbose to see results ! :(

Random.seed!(0)
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
@printf "\nConventional quadrature via QuadGK:\n"
@printf "test realadap for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Aa = realadap(hm,ω,η,tol=tol, verb=1)
@printf "\tAa = "; println(Aa)
TIME(realadap)(hm,ω,η,tol=tol)
@printf "test realadap_lxvm for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Al = realadap_lxvm(hm,ω,η,tol=tol, verb=1)
@printf "\tAl = "; println(Al)
TIME(realadap_lxvm)(hm,ω,η,tol=tol)

#=
@printf "miniquadgk plain...\n"
f(x) = sin(3*x+1)
F(x) = -cos(3*x+1)/3
Ie = F(1)-F(-1)
r = gkrule()
@printf "\t1-seg G-err %.3g, K err %.3g\n" sum(r.gw.*f.(r.x[2:2:end]))-Ie sum(r.w.*f.(r.x))-Ie
seg = applygkrule(f,-1.0,1.0,r)
@printf "\t1-seg (applygkrule) I=%.12g E=%.3g\n" seg.I seg.E
a,b = 0.0,6.0
Ie = F(b)-F(a)
I, E, segs, numevals = miniquadgk(f,a,b)
@printf "\tminiquadgk(%d fevals) err %.3g, \testim E %.3g\n" numevals I-Ie E
I, E, segs, numevals = miniquadgk(f,a,b,rtol=1e-12)
@printf "\tminiquadgk(%d fevals) err %.3g, \testim E %.3g\n" numevals I-Ie E
=#

f(x::Number) = inv(complex(ω,η) - fourier_kernel(hm,x));  # integrand for 1DBZ
@printf "test miniquadgk for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Am, E, segs, numevals = miniquadgk(f,0.0,2π,rtol=tol)    # NOT for timing
# ... since f is not "interpolated". Would need to define f in a function
@printf "\tAm = "; println(Am)
@printf "\t\tabs(Am-Aa)=%.3g\n" abs(Am-Aa)
Am, E, segs, numevals = realmyadap(hm,ω,η,tol=tol)
@printf "test realmyadap (same pars): fevals=%d, nsegs=%d, claimed err=%g\n" numevals length(segs) E
TIME(realmyadap)(hm,ω,η,tol=tol)
rho0=0.5
Ap, E, segs, numevals = realquadinv(hm,ω,η,tol=tol,rho=rho0)
@printf "test realquadinv (same pars): fevals=%d, nsegs=%d, claimed err=%g\n" numevals length(segs) E
@printf "\tAp = "; println(Ap)
@printf "\t\tabs(Ap-Aa)=%.3g\n" abs(Ap-Aa)
TIME(realquadinv)(hm,ω,η,tol=tol,rho=rho0)
print_timer(TIME, sortby=:firstexec)   # otherwise randomizes order!
#plot(segs)

#=
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1
@btime realmyadap(hm,ω,η,tol=tol);
# Before allocated fvals in type-stable way: 842.324 μs (14354 allocations: 515.05 KiB)
# After:  552.549 μs (17 allocations: 45.52 KiB)      good enough
@btime realadap_lxvm(hm,ω,η,tol=tol);
#  551.587 μs (8 allocations: 36.81 KiB)
=#

@printf "1-seg, f = 1/g, try pole subtract... vs 1e-10 tol numer quadr\n"
d = 1e-3
z0 = 0.3+1im*d
g(x) = 2*sin(x-z0)     # complex sin, root @ z0.  Next root dist ~1.8 from [a,b]
gp(x) = 2*cos(x-z0)    # g'
f(x::Number) = 1.0/g(x)
resf0 = 1.0/gp(z0)     # residue of f at its pole
a,b = -1.0,0.5
Im, Em, segs, numevals = miniquadgk(f,a,b,rtol=1e-10);  # right ans, slow
#plot(segs); @gp :- real(z0) imag(z0) "w p pt 1 ps 2 tit 'z_0'"
r = gkrule()
s = applygkrule(f,a,b,r)    # crude version w/ allocs
@printf "\tdumb uncorr 1-seg err %.3g (claimed E %.3g)\n" abs(s.I-Im) s.E
pole(x) = resf0./(x-z0)
sc = applygkrule(x->f(x)-pole(x), a,b,r)  # pole-sub func
Ic = sc.I + resf0*log((b-z0)/(a-z0))      # add exact pole integral
@printf "\tknown-pole corr 1-seg err %.3g (claimed E %.3g)\n" abs(Ic-Im) sc.E
# now fit roots & use resulting extracted residues...
mid, sca = (b+a)/2, (b-a)/2
xj = mid .+ sca*r.x
fj = f.(xj)       # our data on seg
ifj = 1.0./fj   # samples of analytic func
rs, ders = find_near_roots(ifj, r.x)
r0 = mid + sca*rs[1]; gp0 = ders[1]/sca;  # just treat 1st pole (there's only 1)
@printf "\tpole fit loc err %.3g, g'(root) err %.3g\n" abs(r0-z0) abs(gp0-gp(z0))
resfr = 1.0/gp0            # residue of f at its pole
polej = @. resfr/(xj-r0)   # pole vals at nodes  (maybe rewrite in std coords?)
sc = applygkrule(fj.-polej, a,b,r)       # pole-sub vals
Ic = sc.I + resfr*log((b-r0)/(a-r0))     # add exact pole integral
@printf "\tpole-fit corr 1-seg err %.3g (claimed E %.3g)\n" abs(Ic-Im) sc.E
# test self-contained func version...
sc = applypolesub!(ifj,fj,a,b,r)
@printf "\tapplypolesub 1-seg err %.3g (claimed E %.3g)\n" abs(sc.I-Im) sc.E
Ap,E,segs,numevals=adaptquadinv(g,a,b)
@printf "\tadaptquadinv err %.3g (%d segs, claimed E %.3g)\n" abs(Ap-Im) length(segs) E
@printf "\n"

