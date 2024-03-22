# Tester for adaptive Quadratic-Pade plus GCQ variant for 1/sqrt sings 3/22/24.
using Int1DBZ
using Printf
using CairoMakie

# integrand f and answer I:  2D tight-binding case (om=2 band edge; log-sings 0, 2)
eta = 1e-5   # broadening (think of as imag part of om)
tol = 1e-8
G1(om) = 2pi/(1im*sqrt(1-om^2))    # x-integral done, 1D tight-binding model
K(k) = Int1DBZ.ellipkAGM(k)               # local code for complete elliptic integral
G2(om) = 4pi*(K(om/2) - 1im*K(sqrt(1-(om/2)^2)))     # 2D TB model exact

no = 100         # how many omegas to sweep
oms = range(0.0,3.0,no)    # overall energy (need Re om >= 0 for correct sign on Re G... why?)
@printf "2D tight-binding test: %d om vals in [%g %g], eta=%g\n" no minimum(oms) maximum(oms) eta
I=zeros(ComplexF64,no); Ia = zero(I); Ie=zero(I); cts = zeros(Int,no,3)     # store stuff to plot
for (i,om) in enumerate(oms)
    @printf "om=%g...\n" om
    f(y) = G1(om + 1im*eta - cos(y))   # integrand for middle integral (NOT FOR TIMING)
    Ie[i] = G2(om + 1im*eta)                 # analytic (exact) answer    
    Ia[i],Ea,segsa,neva = miniquadgk(f,0,2pi,rtol=tol)      # test conventional adap
    @printf "\texact:\tIe = %.12g + %.12gi\n" real(Ie[i]) imag(Ie[i])
    @printf "\tadap:\tIa = %.12g + %.12gi\t (relerr=%.3g, esterr=%.3g, nsegs=%d, nev=%d)\n" real(Ia[i]) imag(Ia[i]) abs(Ia[i]-Ie[i])/abs(Ie[i]) E length(segsa) neva
    # test adaptive integrator w/ QPade+GCQ option in Int1DBZ module...
    I[i], Ea, s, nev = adaptquadsqrt(f,0.0,2pi,atol=tol,verb=1)
    nsqrt = sum([s.nsqrtsings>0 for s in s])
    cts[i,1] = neva; cts[i,2] = nev; cts[i,3] = nsqrt
    @printf "\ta-QPade:I = %.12g + %.12gi\t (relerr=%.3g, esterr=%.3g, nsegs=%d [nsqrt=%d], nev=%d)\n" real(I[i]) imag(I[i]) abs(I[i]-Ie[i])/abs(Ie[i]) E length(s) nsqrt nev
end

fig = Figure(); ax=Axis(fig[1,1],title="G2 vs omega (eta=$eta)")
scatterlines!(oms, imag(I))
scatterlines!(oms, imag(Ie))
scatterlines!(oms, imag(Ia))
display(fig)
#save("tightbind_test_adaptquadsqrt.png",fig)
