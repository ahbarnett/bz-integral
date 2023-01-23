# benchmark 1D BZ module.
# Barnett 12/26/22. replace @btime w/ TimerOutputs, add math chk 1/20/23
# see btime_Cont1DBZ.jl for old @btime version.

push!(LOAD_PATH,".")
using Cont1DBZ
using OffsetArrays
using StaticArrays
using Printf

using TimerOutputs
TIME=TimerOutput()

using LinearAlgebra
BLAS.set_num_threads(1)       # linalg single-thread for fairness

zsort(z) = sort(z, by = x->reim(x))   # sort C-numbers as in MATLAB meth 'real'
#zsort(z) = sort(z, by = x->(abs(x),angle(x)))  # sort by mag then angle (bad)

x = 2π*rand(1000)             # plain eval targs
η=1e-6; ω=0.5; tol=1e-8;      # integr params
NPTR=30                       # fixed for now

@printf "bench Cont1DBZ...\n"
for M = [8,32,128], T in (ComplexF64, SMatrix{1,1,ComplexF64,1}, SMatrix{5,5,ComplexF64,25})
    @timeit TIME string(T) begin
    local hm = OffsetVector(randn(T,2M+1),-M:M)      # h(x)
    local hm = (hm + conj(reverse(hm)))/2                     # make h(x) real
    @timeit TIME @sprintf("M=%d (eval at %d targs)",M,length(x)) begin
        for i=1:10   # samples
            TIME(evalh_ref)(hm,x)
            TIME(evalh_wind)(hm,x)
            TIME(evalh)(hm,x)
            TIME(fourier_kernel)(hm,x)    # note uses array version
        end
        T<:Number || continue # for now skip root-finding for matrix coefficients
        coeffs = reverse(hm.parent)
        @timeit TIME @sprintf("root-finding (matrix size 2M+1=%d)",2M+1) begin
            r = TIME(roots)(coeffs)
            r2 = TIME(roots_best)(coeffs)
        end
        @printf "roots max diff = %.3g\n" maximum(abs.(zsort(r).-zsort(r2)))
        @timeit TIME @sprintf("quadr for ω=%g η=%g tol=%g",ω,η,tol) begin
            A = TIME(realadap)(hm,ω,η,tol=tol)
            AL = TIME(realadap_lxvm)(hm,ω,η,tol=tol)
            AI = @timeit TIME @sprintf("imshcorr(NPTR=%d)",NPTR) imshcorr(hm,ω,η,N=NPTR)
        end
        println(A,'\n',AL,'\n',AI)   # check integrals match
    end
    end
end
print_timer(TIME, sortby=:firstexec)   # use sortby otherwise randomizes order!
#print_timer(TIME, sortby=:firstexec, allocations=false, compact=true)
# reset_timer!(TIME)
