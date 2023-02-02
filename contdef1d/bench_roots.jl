# compare benchmarks for polynomial root-finding (complex-float64 coeffs),
# various pkgs. Barnett 1/19/23.

# cf timing demos at https://github.com/andreasnoack/FastPolynomialRoots.jl

push!(LOAD_PATH,".")
import Cont1DBZ        # import not using, to keep namespaces separate
import AMRVW         # native Julia AMVW, claimed not as fast as Fortran
import Polynomials
P = Polynomials    # abbrev
import FastPolynomialRoots   # sadly this *redefines* P.roots() preventing test
import PolynomialRoots    # Giordano low-M astro pkg (a la Skowron'12, NumRec)
using Printf

if true
    @printf "correctness tests...\n"
    # real coeffs case, roots should be
    r = [1. 2. 3. 4.]
    p4 = [24.0, -50.0, 35.0, -10.0, 1.0]  # (x-1) * (x-2) * (x-3) * (x-4)
    r1 = Cont1DBZ.roots(reverse(p4))
    println(maximum(abs.(r' .- sort(real(r1)))))
    r2 = AMRVW.roots(p4)
    println(maximum(abs.(r' .- sort(real(r2)))))
    r3 = P.roots(P.Polynomial(p4))           # Polynomials pkg
    println(maximum(abs.(r' .- sort(real(r3)))))
    r4 = FastPolynomialRoots.rootsFastPolynomialRoots(p4)
    println(maximum(abs.(r' .- sort(real(r4)))))
    r5 = PolynomialRoots.roots(p4)
    println(maximum(abs.(r' .- sort(real(r5)))))
    # complex coeffs case, roots now should be 1i,2i,3i,4i...
    p4 = [24.0, 50.0*im, -35.0, -10.0*im, 1.0]  # above but w/ powers of i
    r1 = Cont1DBZ.roots(reverse(p4))
    println(maximum(abs.(r' .- sort(imag(r1)))))
    r2 = AMRVW.roots(p4)
    println(maximum(abs.(r' .- sort(imag(r2)))))
    r3 = P.roots(P.Polynomial(p4))           # Polynomials pkg
    println(maximum(abs.(r' .- sort(imag(r3)))))
    r4 = FastPolynomialRoots.rootsFastPolynomialRoots(p4)
    println(maximum(abs.(r' .- sort(imag(r4)))))
    r5 = PolynomialRoots.roots(p4)
    println(maximum(abs.(r' .- sort(imag(r5)))))
end

# timing tests (@elapsed sufficient and less annoying than @btime)...

using LinearAlgebra
BLAS.set_num_threads(1)       # linalg single-thread for fairness, also faster!

@printf "\nspeed tests...\ndegr d\t\tmyroots\tAMRVW\tP.r,not\tFastPR\tPR.r\t(times in ms)\n"
#for d = 1 .<< (4:10)        # sizes (poly degrees) to test
for d in 100:100:200
    @printf "%d\t\t" d
    p = randn(ComplexF64,d)    # complex coeffs
    rp = reverse(p)           # precompute
    pp = P.Polynomial(p)
    t_ns1 = @elapsed local r1 = Cont1DBZ.roots(rp)   # my companion->eigvals M^3
    t_ns2 = @elapsed local r2 = AMRVW.roots(p)   # native jl AMVW
    t_ns3 = @elapsed local r3 = P.roots(pp)      # Polynomials pkg M^3 ... except this is overwritten by using FPR ... ugh!
    t_ns4 = @elapsed local r4 = FastPolynomialRoots.rootsFastPolynomialRoots(p) # Fortran AMVW
    t_ns5 = @elapsed local r5 = PolynomialRoots.roots(p)  # astro pkg
    # was benchmarktools, too annoyingly slow...
    #    t_ns1 = minimum((@benchmark r1 = Cont1DBZ.roots($rp)).times)   # my companion->eigvals M^3
    #    t_ns2 = minimum((@benchmark r2 = AMRVW.roots($p)).times)   # native jl AMVW
    #    t_ns3 = minimum((@benchmark r3 = P.roots($pp)).times)      # Polynomials pkg M^3
    #    t_ns4 = minimum((@benchmark r4 = FastPolynomialRoots.rootsFastPolynomialRoots($p)).times)       # Fortran AMVW
    #    t_ns5 = minimum((@benchmark r5 = PolynomialRoots.roots($p)).times)  # astro pkg
    s=1.0e-3 # 1e6           # s -> ms factor
    @printf "%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" t_ns1/s t_ns2/s t_ns3/s t_ns4/s t_ns5/s
    if false
        @printf "errs:\t\t%.3g" maximum(abs.(r1))-maximum(abs.(r2))
        @printf "\t%.3g" maximum(abs.(r1))-maximum(abs.(r3))
        @printf "\t%.3g" maximum(abs.(r1))-maximum(abs.(r4))
        @printf "\t%.3g\n" maximum(abs.(r1))-maximum(abs.(r5)) # d>200 fails!
    end
end

# julia -t1, BLASthreads=1: (& I wrote in the Poly.r by hand due to FPR dumb):
#degr d		myroots	AMRVW	Poly.r	FastPR	PR.r	(times in ms)
#16		0.0833	0.0819	0.0799	0.0782	0.0113
#32		0.32	0.307	0.361	0.305	0.0383
#64		1.62	1.15	1.65	1.09	0.153
#128		13.5	4.18	12.6	4.08	0.572
#256		76	15.6	76.8	15.1	3.99
#512		657	58.4	597	56.4	109
#1024		2.62e+03 224	2.53e+03 213	434

# Conclusions:
# 1) native jl AMWVR pkg is within few % of Fortran and FPR
# 2) FPR dumbly overwrites P.roots method so you can't compare together
# 3) PR is 8x faster than any other, but dies w/ O(1)/NaN errs for d>=200.
# Suggest PR for d<150 or so, but AMRVW for d greater.
# 4) for vector (eg n=3) case for Jason, none of this will matter; need to
#    generalize to block companion eigvals, or iterative Skowron'12 NEVP invent


# OLDER tests...

# multithreaded julia, BLASthreads=1: (same for BLASthreads=8):
#degr d		myroots	AMRVW	Poly.r	FastPR	(times in ms)
#8		0.0176	0.0224	0.0207	0.0207
#32		0.311	0.308	0.301	0.295
#128		10.3	4.34	4.05	4.08
#512		634	60.4	56.7	56.4
#2048		1.36e+04 892	819	817

# julia -t1:
#degr d		myroots	AMRVW	Poly.r	FastPR	(times in ms)
#16		0.0802	0.0886	0.0859	0.0844
#32		0.325	0.306	0.304	0.308
#64		1.6	1.15	1.1	1.11
#128		13.3	4.23	4.11	4.08
#256		83.2	15.6	15.2	15.1
#512		623	59.7	57.2	57.4
#1024		2.53e+03 227	214	215

