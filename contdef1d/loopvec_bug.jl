# demo a potential bug in LoopVectorization.jl
# involving dependent tuple assignment. In the end could not get x = x0 to avx!
# Barnett 12/29/22
using LoopVectorization
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1

t = 0.7  # arb real
s,c=sincos(t)
x,y=1.0,0.0        # initialize
# we now repeatedly multiply [x;y] by t-rotation matrix [c,-s;s,c]
n = 1000
sn,cn=sincos(n*t)   # here's the known addition formula answer (rotation)
#function rot!(x,y,c,s,n)
    @inbounds @fastmath for i=1:n
        global x,y = c*x-s*y, s*x+c*y
    end
#end
#@btime rot!(x,y,c,s,n)
println("inbounds fastmath error:\t",maximum(abs.([x-cn,y-sn])))

x,y=1.0,0.0
@avx for i=1:n
    # all fails to compile with @avx
#    global x,y = c*x-s*y, s*x+c*y
    local x0 = c*x-s*y
    local y0 = s*x+c*y
    global x = x0
    global y = y0
end
println("avx error:\t\t\t",maximum(abs.([x-cn,y-sn])))

