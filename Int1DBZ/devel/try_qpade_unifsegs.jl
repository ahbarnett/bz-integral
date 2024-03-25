# try combining quadratic pade with GCQ for auto-sqrt handling on 1 seg.
# Then, uniform segs, then fully adaptive test.
# Barnett 2/22/24, restart post-Bonaire 3/20/24.
using Int1DBZ
using Printf
using CairoMakie

# integrand f and answer I:  2D tight-binding case (om=2 band edge; log-sings 0, 2)
om = 0.7      # overall energy (need Re om >= 0 for correct sign on Re G... why?)
eta = 1e-5   # broadening (think of as imag part of om)
G1(om) = 2pi/(1im*sqrt(1-om^2))    # x-integral done, 1D tight-binding model
f(y) = G1(om + 1im*eta - cos(y))   # integrand for middle integral (NOT FOR TIMING)
K(k) = Int1DBZ.ellipkAGM(k)             # local code for complete elliptic integral
G2(om) = 4pi*(K(om/2) - 1im*K(sqrt(1-(om/2)^2)))     # 2D TB model
Ie = G2(om + 1im*eta)                 # analytic (exact) answer
@printf "2D tight-binding test: om=%g, eta=%g\n" om eta

# test conventional adap
Ia,E,segs,nev = miniquadgk(f,0,2pi,rtol=1e-10)
@printf "exact:\tIe = %.12g + %.12gi\n" real(Ie) imag(Ie)
@printf("adap:\tIa = %.12g + %.12gi\t (relerr=%.3g, esterr=%.3g, nsegs=%d, nev=%d)\n",
        real(Ia),imag(Ia),abs(Ia-Ie)/abs(Ie),E,length(segs),nev)
fig = Figure(); ax=Axis(fig[1,1],title="miniquadgk segs (om=$om, eta=$eta)")
showsegs!(segs)
#display(fig)

if false   # test by-hand plain composite quad, uniform segments over [0,2pi)
ns = 100
L = 2pi/ns
r = gkrule(); p = length(r.x)
Iu = 0.0; Eu = 0.0;
for i=1:ns
    s = applygkrule(f,(i-1)*L,i*L,r)
    Iu += s.I; Eu += s.E
end
@printf("unif:\tIu = %.12g + %.12gi\t (relerr=%.3g, esterr=%.3g, nsegs=%d, nev=%d)\n",
    real(Iu),imag(Iu),abs(Iu-Ie)/abs(Ie),Eu,ns,ns*p)
end

if false    # warm-up dev implementation: Q-Pade uniform segments over [0,2pi)
ns = 9
r = gkrule(); p = length(r.x)    # gk allows err est for non-qpade segs
tolg = 1e-10; tolg2 = 1e3*tolg     # tol for GCQ (good and GK-type less good)
verbg = 0                        # text output for GCQ?
L = 2pi/ns
Iq = 0.0; Eq = 0.0; nev=0          # answer, error est, f-eval counter
for i=1:ns
    a = (i-1)*L; b=i*L
    @printf "qpade seg [%g,%g]..." a b
    mid, sca = (b+a)/2, (b-a)/2
    fj = f.(mid .+ sca*r.x)               # first set of p expensive f evals
    nev += p
    zsing,dzsing = qpade_sqrtsings(fj,r.x,rho=exp(1))  # sing locs scaled to [-1,1]
    println("\tsqrt sings: ", mid .+ sca*zsing)
    #println("\t\td/dz of sings: ",sca*dzsing)
    nq = length(zsing)
    if nq==0                        # no sqrt sings near; use plain
        s = applygkrule(fj,a,b,r)
        Iq += s.I; Eq += s.E
    elseif nq==1           # found one sqrt-sings
        z0 = zsing[1]      # get it
        pg = p         # poly degree of func set (poly + poly/sqrt), send to GCQ
        fs(x::Number) = reduce(vcat, x^k.*[1, reim(1/sqrt(x-z0))...] for k=0:pg)
        xg, wg, _ = genchebquad(fs,-1.0,1.0,tolg;verb=verbg)       # build a GCQ
        ng = length(xg)
        @printf("\tGCQ built (z0=%.10g+%.10gi, tol=%g, %d nodes)\n",
                 real(z0),imag(z0),tolg,ng)
        fg = f.(mid .+ sca*xg)                 # do expensive f evals
        xg2, wg2, _ = genchebquad(fs,-1.0,1.0,tolg2;verb=verbg)  # build a worse GCQ
        ng2 = length(xg2)
        @printf("\thalf-order GCQ built (z0=%.10g+%.10gi, tol=%g, %d nodes)\n",
                real(z0),imag(z0),tolg2,ng2)
        fg2 = f.(mid .+ sca*xg2)                 # more f evals for err estim
        nev += ng2
        Is = sca*sum(fg .* wg)
        Is2 = sca*sum(fg2 .* wg2)
        Iq += Is
        Eq += abs(Is-Is2)    # err estim
        println("\t\tqpade estim err: ", abs(Is-Is2))
    else
        @printf "\tnq=%d > 1: no scheme for multiple sqrt-sings yet!\n" nq
    end
end
@printf("QPade:\tIq = %.12g + %.12gi\t (relerr=%.3g, esterr=%.3g, nsegs=%d, nev=%d)\n",
    real(Iq),imag(Iq),abs(Iq-Ie)/abs(Ie),Eq,ns,nev)
end

# test adaptive integrator w/ QPade+GCQ option in Int1DBZ module...
atol = 1e-8
Ia, Ea, sa, neva = adaptquadsqrt(f,0.0,2pi,atol=atol,verb=1)
nsqrt = sum([s.nsqrtsings>0 for s in sa])
@printf("a-QPade:Ia = %.12g + %.12gi\t (relerr=%.3g, esterr=%.3g, nsegs=%d [nsqrt=%d], nev=%d)\n",
    real(Ia),imag(Ia),abs(Ia-Ie)/abs(Ie),Ea,length(sa),nsqrt,neva)
ax=Axis(fig[2,1],title="adap QPade+GCQ: special segs shown green")
showsegs!(sa)
display(fig);
#save("tightbind_test_qpade.png",fig)




