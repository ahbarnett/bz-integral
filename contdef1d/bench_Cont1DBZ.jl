# benchmark 1D BZ module. Barnett 12/26/22

push!(LOAD_PATH,".")
using Cont1DBZ
using OffsetArrays
using Printf

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1  # for btime only (<0.1 no help)

using LinearAlgebra
BLAS.set_num_threads(1)       # linalg single-thread for fairness

x = 2π*rand(1000)
η=1e-6; ω=0.5; tol=1e-8;
NPTR=30

for M = [1,10,100]
    @printf "bench Cont1DBZ with M=%d.\n\tEval at %d targs...\n" M length(x)
    hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # h(x)
    hm = (hm + conj(reverse(hm)))/2                     # make h(x) real
    @printf "evalh_ref:\t"
    @btime evalh_ref($hm,$x)
    @printf "evalh_wind:\t"
    @btime evalh_wind($hm,$x)
    @printf "evalh:    \t"
    @btime evalh($hm,$x)        # why so many allocs & RAM?
    t_ns = minimum((@benchmark evalh($hm,$x)).times)
    @printf "evalh %g G mode-targs/sec\n" (2M+1)*length(x)/t_ns
    
    @printf "\tQuadr meths:\nrealadap ω=%g η=%g tol=%g:  " ω η tol
    @btime realadap($hm,ω,η,tol=tol)
    @printf "\tQuadr meths:\nrealadap_lxvm ω=%g η=%g tol=%g:  " ω η tol
    @btime realadap_lxvm($hm,ω,η,tol=tol)
    @printf "roots (matrix size 2M+1=%d):  " 2M+1
    coeffs = reverse(hm.parent)
    @btime roots($coeffs)
    @printf "imshcorr same ω and η as above, NPTR=%d:   " NPTR
    @btime imshcorr($hm,ω,η,N=NPTR)
end
