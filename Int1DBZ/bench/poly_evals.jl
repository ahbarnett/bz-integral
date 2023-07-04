# showing how splatting unknown-length Vector slows down poly eval by
# 1000x :(

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1

# Poly eval (from SGJ's 2019 JuliaCon talk)
# see 27 mins into: https://www.youtube.com/watch?v=mSgXWpvQEHE
horner(x::Number)=zero(x)
horner(x::Number,p1::Number)=p1
horner(x::Number,p1::Number,p...)=muladd(x,horner(x,p...),p1)  # labeled splat p
# handle coeff vector rather than list of args, by splat...
#horner(x::Number,c::Vector)=horner(x,c...)   # no, slows it down :(
# handle array arg x...
horner(x::AbstractArray,args...) = map(y -> horner(y,args...), x)
# Timing expts...
#a = rand(32); x = rand()
#julia> @btime horner($x,$a);
#  17.944 μs (654 allocations: 13.97 KiB)      <- splatting each time = bad
#julia> @btime horner($x,$(a...));
#  2.083 ns (0 allocations: 0 bytes)

p = rand(ComplexF64,15)   # coeffs

@btime horner(0.7,($p)...);   # 5us 7kB    <- 1000x slower!
@btime horner(0.7,$(p...));  # 2.3 ns  0bytes    <-fastest, compiles for degree?
@btime Base.evalpoly(0.7,$p);  # 14 ns 0bytes ... penalty for unknown length
@btime Base.evalpoly(0.7,$(tuple(p...)));   # 5ns 0bytes

# see https://discourse.julialang.org/t/improving-computational-performance/63210
