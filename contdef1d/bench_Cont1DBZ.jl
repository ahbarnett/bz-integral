# benchmark 1D BZ module.
# Barnett 12/26/22. replace @btime w/ TimerOutputs, add math chk 1/20/23
# see btime_Cont1DBZ.jl for old @btime version.
# 1/26/23: decaying coeffs -> fewer near-zeros -> realadap faster :(

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
η=1e-4; ω=0.5; tol=1e-8;      # integr params
NPTR=30                       # fixed for now
n = 5                         # non-1 matrix size to test

@printf "bench Cont1DBZ...\n"
#for M = [8,32,128], T in (ComplexF64, SMatrix{1,1,ComplexF64,1}, SMatrix{5,5,ComplexF64,25})   # too many cases for now :)
 for M = [10], T in (ComplexF64, SMatrix{n,n,ComplexF64,n^2})
    @timeit TIME @sprintf("Type: %s",string(T)) begin
        local hm = OffsetVector(randn(T,2M+1),-M:M)   # h(x)
        local hmconj = OffsetVector([hm[m]' for m in -M:M], -M:M)   # ugh!!
        hm = (hm + reverse(hmconj))/2                 # h(x) hermitian if x Re
        hm = OffsetVector([exp(-0.5*abs(m))*hm[m] for m in -M:M], -M:M) # decay
        #@printf "check hm has Herm symm : %.3g\n" norm(hm[M]'-hm[-M],Inf)
        @timeit TIME @sprintf("M=%d (eval at %d targs)",M,length(x)) begin
            for i=1:10   # samples
                TIME(evalh_ref)(hm,x)
                TIME(evalh_wind)(hm,x)
                TIME(evalh)(hm,x)
                TIME(fourier_kernel)(hm,x)    # note uses array version
            end
            if T<:Number # for now skip root-finding for matrix coefficients
                coeffs = reverse(hm.parent)
                @timeit TIME @sprintf("root-finding (mat size 2M+1=%d)",2M+1) begin
                    r = TIME(roots)(coeffs)
                    r2 = TIME(roots_best)(coeffs)   # *** failing for M=32..50
                end
                @printf "roots max diff = %.3g\n" maximum(abs.(zsort(r).-zsort(r2)))
            end
            # some quadr done for n=1 and n>1...
            @timeit TIME @sprintf("quadr for ω=%g η=%g tol=%g",ω,η,tol) begin
                A = TIME(realadap)(hm,ω,η,tol=tol)
                AL = TIME(realadap_lxvm)(hm,ω,η,tol=tol)
                if T<:Number
                    AI = @timeit TIME @sprintf("imshcorr(NPTR=%d)",NPTR) imshcorr(hm,ω,η,N=NPTR)
                else; AI=NaN; end
                AD = TIME(discresi)(hm,ω,η, verb=0)
            end
            println("A =",A,"\nAL=",AL,"\nAI=",AI,"\nAD=",AD,"\n")   # eyeball integrals match
        end
    end
end
print_timer(TIME, sortby=:firstexec)   # use sortby otherwise randomizes order!
#print_timer(TIME, sortby=:firstexec, allocations=false, compact=true)
# reset_timer!(TIME)
