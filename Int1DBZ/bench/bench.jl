# a speed test for scalar case of new quadrature meths. run with
# julia -t1 --track-allocation=user --project=.

using Int1DBZ
using OffsetArrays
using Printf
using Random.Random


M=20            # max mag Fourier freq index (200 to make fevals slow)
η=1e-6; ω=0.5; tol=1e-7;  # 1e-8 too much for M=200 realadap to handle :(
verb = 1
Random.seed!(0)         # set up 1D BZ h(x) for denominator
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # F-coeffs of h(x)
hm = (hm + conj(reverse(hm)))/2                     # make h(x) real for x Re

#A,E,segs,numevals = realquadinv(hm,ω,η; tol=tol);
A,E,segs,numevals = realmyadap(hm,ω,η; tol=tol);
@printf "Ar=%g+%gi, E=%.3g, %d segs, fevals/(2n+1)=%d\n" real(A) imag(A) E length(segs) Int(numevals/15)

#=    # this gives a useless graph in web browser... no line numbers
using Profile, PProf
Profile.Allocs.clear()
Profile.Allocs.@profile (for i=1:1000
                         Ar,E,segsr,numevals = realquadinv(hm,ω,η; tol=tol);
                         end )
PProf.Allocs.pprof()
=#
