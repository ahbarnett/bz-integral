# performance (robustness, speed, err) test of adap-QPade+GCQ, in 2D
# random Fourier series setting (warm up via M=1 TB model).   Barnett 3/24/24.
using Int1DBZ
using Printf
using OffsetArrays     # for F-series
using StaticArrays     # only for n>1
using LinearAlgebra
BLAS.set_num_threads(1)
using Random.Random
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1
using GLMakie

# middle integrand f(y) := \int_0^{2π} Tr (om+i.eta-H(x,y))^{-1} dx
function fmid(y::Number,H::AbstractArray,om,eta;tol=1e-6,rootmeth="PR")
    Hm = [fourier_kernel(H[:,m2],y) for m2 in axes(H,2)]   # eval 1d F coeffs (OV)
    Ap, Ep, segsp, nep = realquadinv(Hm,om,eta, tol=tol,rootmeth=rootmeth)
    #println(nep)
    Ap
end
function fmidnaive(y::Number,H::AbstractArray,om,eta;tol=1e-6)   # use naive quadgk
    Hm = [fourier_kernel(H[:,m2],y) for m2 in axes(H,2)]   # eval 1d F coeffs (OV)
    Ap, Ep, segsp, nep = realmyadap(Hm,om,eta, tol=tol)
    Ap
end

if 0==1   # warm up: tight-binding (n=1 scalar, freq M=1):  H(x,y) = cos x + cos y
H = OffsetArray(zeros(3,3),-1:1,-1:1); H[0,1]=H[0,-1]=H[1,0]=H[-1,0]=0.5
# note i (row index) for inner x ordinate; j (col) for middle y ordinate
eta=1e-5; om=0.6   # fix for now
tol=1e-6         # overall tol
itol = 1e-2*tol      # tol for performing inner (x) integral
@printf "2D tight binding G_2 test: om=%g eta=%g (tol=%g):\n" om eta tol

#tobj = @benchmark fmid(1.0,om,eta)
# compare analytic
G1(om) = 2pi/(1im*sqrt(1-om^2))    # x-integral done, 1D tight-binding model
K(k) = Int1DBZ.ellipkAGM(k)        # local code for complete elliptic integral
G2(om) = 4pi*(K(om/2) - 1im*K(sqrt(1-(om/2)^2)))   # 2D TB model exact (Re om>=0)
Ie = G2(om+1im*eta)
fana(y,om,eta) = G1(om + 1im*eta - cos(y))   # integrand for middle integral
y=1.4;
@printf "\tintegrand fmid(%g) err:      %.3g\n" y abs(fmid(y,H,om,eta)-fana(y,om,eta))
@printf "\tintegrand fmidnaive(%g) err: %.3g\n" y abs(fmidnaive(y,H,om,eta)-fana(y,om,eta))
f2(y) = fmid(y,H,om,eta,tol=itol)      # single-argument middle integrand
Ia,Ea,sa,neva = miniquadgk(f2,0.0,2pi,atol=tol)
@printf "adap:\tIa=%.12g+%.12gi  \terr vs anal %.3g,\t%d evals\n" real(Ia) imag(Ia) abs(Ia-Ie) neva
I, E, s, nev = adaptquadsqrt(f2,0.0,2pi,atol=tol,verb=1)
@printf "aQPade:\tI=%.12g+%.12gi  \terr vs anal %.3g,\t%d evals\n" real(I) imag(I) abs(I-Ie) nev
end

# general F-series (adap from 1D timingtable of 2023)...
n=1             # matrix size for H (1 is scalar)
M=10            # max mag Fourier freq index: (2M+1)^2 coeffs in 2D
mtail = 1e-2    # how small exp decay of coeffs gets to by m=+-M in either axis
mlist = -M:M    # matrix, OV of SA's version, some painful iterators here...
decayrate = log(1/mtail)/M
am = OffsetVector(exp.(-decayrate*abs.(mlist)), mlist)  # ampl decay
Random.seed!(0)         # set up 2D BZ H(x,y) as Fourier coeffs
H = OffsetArray([SMatrix{n,n}(am[m1]*am[m2] * randn(ComplexF64,(n,n)))
                for m1 in mlist, m2 in mlist], mlist, mlist)     # fill 2D OA
Hconj = OffsetArray([H[m1,m2]' for m1 in mlist, m2 in mlist], mlist, mlist)
H = (H + reverse(Hconj))/2   # reverse flips both indices: H herm for real args
eta=1e-5; om=0.6   # fix for now
tol=1e-6
itol = 1e-2*tol      # tol for performing inner (x) integral
@printf "Test rand 2D F-series: n=%d M=%d ω=%g η=%g tol=%g itol=%g...\n" n M om eta tol itol
f2(y) = fmid(y,H,om,eta,tol=itol,rootmeth="F")   # single-argument middle integrand
f2naive(y) = fmidnaive(y,H,om,eta,tol=itol)   # same but use miniquadgk for inner
@printf "\tintegrand fmid vs naive(%g) diff %.3g\n" y abs(f2naive(y)-f2(y))
Ia,Ea,segsa,neva = miniquadgk(f2,0,2pi,atol=tol)       # test naive adap middle
@printf("adap:\tIa= %.12g + %.12gi\t (esterr=%.3g, nsegs=%d, nev=%d)\n",
    real(Ia),imag(Ia),Ea,length(segsa),neva)
I, E, s, nev = adaptquadsqrt(f2,0.0,2pi,atol=tol,verb=1)  # test adap+QPade+GCQ 
nsqrt = sum([s.nsqrtsings>0 for s in s])
@printf("aQPade:\tI = %.12g + %.12gi\t (reldiff=%.3g, esterr=%.3g, nsegs=%d [nsqrt=%d], nev=%d)\n",
    real(I),imag(I),abs(I-Ia)/abs(Ia),E,length(s),nsqrt,nev)
if verb>0 fig = Figure()     # examine segs (green shows special qpade+GCQ used)
    ax=Axis(fig[1,1],title="miniquadgk segs (om=$om, eta=$eta): nev=$neva")
    showsegs!(segsa)
    ax2=Axis(fig[2,1],title="a-QPade+GCQ segs: nev=$nev (nsqrtsegs=$nsqrt)")
    showsegs!(s)
    display(fig)
end
