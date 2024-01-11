# play with complete elliptic integral K in complex plane
# Barnett 1/9/24

# Note complex arg only just implemented, in EllipticFunctions.jl
# https://discourse.julialang.org/t/elliptic-functions-of-complex-argument/51876
# Docs: https://stla.github.io/EllipticFunctions.jl/dev/
using EllipticFunctions

using GLMakie
using Int1DBZ

dx=0.02; L=2              # resolution, half-field-of-view
gx = range(-L,L,step=dx); gy = range(-L,L,step=dx)     # plot grid
zg = kron(gx,ones(size(gy'))) .+ 1im*kron(ones(size(gx)),gy')  # C grid
Kg = ellipticK.(zg)     # arg is m = k^2 the squared "modulus"
C = Makie.to_color(z2color.(Kg))
fig,ax,h = heatmap(gx,gy,C, axis=(aspect=DataAspect(),))
ax.xlabel=L"Re $m$"; ax.ylabel=L"Im $m$"
GLMakie.activate!(title = "ellipticK in C-plane of m=k^2")
display(fig)

om = -0.7       # check |om|<2 real in band
k = om/2
K(k) = ellipticK(k^2)   # wrapper with arg modulus k to EllipticFunctions
abs(K(1/k) - sqrt(k^2)*(K(k)-1im*K(sqrt(1-k^2))))   # Legendre connection, works

KA(k) = ellipkAGM(k)   # let's compute elliptic K ourselves! (see ellipk.jl)
k=0.7; u=KA(k); v=K(k); u,v, abs(u-v)
n=100000;   # speed test
#maximum([AGM(sqrt(1-Complex(k)^2),1)[2] for k in 2*rand(n)+1im*rand(n)])  # max its
@time for i=1:1000000; ellipticK(2.0*rand()+1im*rand()); end
ellipticKA(m) = KA(sqrt(m))
@time for i=1:1000000; ellipticKA(2.0*rand()+1im*rand()); end
# we're 10x faster than EF lib!
println("max abs diff: ",maximum([abs(KA(k)-K(k)) for k in 4*(rand(n).-0.5)+4im*(rand(n).-0.5)]))
# See my post:  https://github.com/stla/EllipticFunctions.jl/issues/34
# for k>1 real, Im KA>0 but Im K<0; differ on this cut (KA has k UHP lim, K has LHP)
# for k<1 real, KA=K, good.
k=0.7; u=KA(1/k); v=k*(KA(k)+1im*KA(sqrt(1-k^2))); u,v,abs(u-v)  # Legendre conn
println("K(1/k)\t\t\t\t\t\tconn. formula (19.7.3)\t\t\t\tabs diff")
for k=0.5*cis.(2pi*(0:7)/8); u=KA(1/k); v=sqrt(k^2)*(KA(k)-1im*sign(imag(k^2))*KA(sqrt(1-k^2))); println(u,"\t",v,"\t",abs(u-v)); end
# still off for k>0 (pos real case)

k=1.7+1e-10im; u=KA(k); v=K(k); u,v, abs(u-v)  # shows KA takes above cut lim

# note weirdness (Kahan famous 1985 paper dislikes) for IEEE zsqrt:
sqrt(-1.0 + 0im)
sqrt(-1.0 - 0im)
# oh dear! the sign bit of the 0 im is being used to determine the branch cut.
# apparently std in C++ math, IEEE 754.
# https://github.com/JuliaLang/julia/issues/21259
# More confusing is:
Complex(-1.0)        # im sign +0
Complex(-1.0)^2      # im sign -0
# toy for G_1(om) Greens func on Re axis:  already gets correct branch cut for om<1 and >1
[1/(1im*sqrt(1-Complex(om)^2)) for om in -3.0:3.0]    # signs are as Im om = 0^+ lim.

1/Complex(1,0)   # im sign -0,  neat way to get it



