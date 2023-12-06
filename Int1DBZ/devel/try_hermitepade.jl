# try Hermite-Pade (really: quadratic Pade of Shafer '74, working with
# function values on std interval, as in robust Pade of Gonnet et al '13),
# for fitting rational or inv-sqrt singular functions.
# Based on fit_invsqrt.jl
# Barnett 12/1/23
using Int1DBZ
using GLMakie
using LinearAlgebra
using FastGaussQuadrature
verb=1

d = 1e-2         # imag dist of sing
println("try_hermitepade...     sing dist d=",d)
z0 = 0.3+1im*d    # sing loc
zz = -0.5-0.2im    # nearby zero loc
#f(x::Number) = sin(x-zz)/sin(x-z0)    # zero & pole, not a rat
#f(x::Number) = (x-zz)/(x-z0)          # (1,1) rat
zz = NaN           # no zero. control freq of sin here...
#f(x::Number) = 0.7 + x^2/3 + sqrt(1im*sin((x-z0)/2))  # +1/2 pow sing, branch up
f(x::Number) = 0.7 + x/3 + 1/sqrt(1im*sin((x-z0)/2))  # -1/2 pow sing, recip

r = gkrule(); xj = r.x; N=length(xj)    # all of GK rule on [-1,1]
#N = 15; xj = [cos(pi*j/(N-1)) for j in 0:N]
#N=15; xj,~ = gausslegendre(N)
fj = f.(xj)    # data
t = range(-1.0,1.0,1000)     # for cont curve plot & max estim on [-1,1]
ft = f.(t)                   # true vals on dense grid on [-1,1]
if verb>0                    # two-panel plot of f on interval & C-plane
    fig=Figure();
    ax,=scatter(fig[1,1],xj,real.(fj), label=L"Re $f_j$")       # nodes
    scatter!(xj,imag.(fj), label=L"Im $f_j$")
    lines!(t,real.(ft))           # curve
    lines!(t,imag.(ft))
    ax.xlabel=L"t"
    axislegend()
    dx=0.02
    gx = range(-2,2,step=dx); gy = range(-1,1,step=dx)
    zg = kron(gx,ones(size(gy'))) .+ 1im*kron(ones(size(gx)),gy')
    fg = f.(zg)      # eval
    ax2,h=heatmap(fig[2,1],gx,gy,abs.(fg), axis=(aspect=DataAspect(),))
    h.colorrange=10.0.^[-1,1]; h.colormap=:jet; h.colorscale=log10;
    Colorbar(fig[2,2],h, label=L"|f|")
    scatter!(real.(xj),imag.(xj), label=L"x_j", color=:black, markersize=5)
    scatter!(real.(z0),imag.(z0), label="sing", marker='*')
    isnan(zz) || scatter!(real.(zz),imag.(zz), label="zero")
    axislegend()
    display(fig)
end
r,gpr = find_near_roots(1.0./fj, xj, rho=exp(1.0), meth="PR")
println("\tpole loc err in find_near_roots on 1/f = ",abs(r[1]-z0))
# NB fails when zero nearby. But its speed in inner BZ setting is key

p = N-1      # degree for single poly
V = [x^i for x in xj, i in 0:p]    # Vandermonde
co = V\fj
fat = [evalpoly(x,co) for x in t]          # f's approximant on t grid
println("\tpoly(",p,") fit f: max resid = ",norm(fat .- ft,Inf))
# fails when nearby sing, obvi

p1 = N÷2-1     # degree of p and q. Do Gonnet-Güttel-Trefethen '13 fit via SVD
V1 = V[:,1:p1+1]; V1 = [V1 diagm(fj)*V1]
S = svd(V1)
println("\tm=1: sing vals = ", S.S)
nullco = S.Vt[end,:]'              # ' = conj transp, needed!
cop = nullco[1:p1+1]; coq = nullco[p1+2:end]    # extract p,q poly coeffs
fat1 = [-evalpoly(x,cop)/evalpoly(x,coq) for x in t]  # grid eval rat -p/q
println("\trat(",p1,",",p1,") fit f: max resid = ",norm(fat1 .- ft,Inf))
# good for nearby pole with nearby zero

p2 = N÷3-1      # degree of p,q,r. Do Fasondini '18 Hermite-Pade fit via SVD
V2 = V[:,1:p2+1]; V2 = [V2 diagm(fj)*V2 diagm(fj.^2)*V2]
S = svd(V2)
println("\tm=2: sing vals = ", S.S)
nullco = S.Vt[end,:]'              # ' = conj transp, needed!
cop = nullco[1:p2+1]; coq = nullco[p2+2:2p2+2]; cor = nullco[2p2+3:end]
#cor, cop = cop, cor              # flip equiv to m=-2, failed
function psi(z::Number)     # approximant, psi, two branches (uses coeffs)
    q = evalpoly(z,coq)
    r = evalpoly(z,cor)
    D = q^2 - 4*r*evalpoly(z,cop)       # discriminant
    (-q+sqrt(D))/2r, (-q-sqrt(D))/2r    # quadratic formula, two branches
end
#fat2 = [psi(x)[1] for x in t]  # grid eval one of psi branches
fat2 = [psi(x)[ abs(psi(x)[1]-f(x))<abs(psi(x)[2]-f(x)) ? 1 : 2 ] for x in t] # best branch
println("\trat(",p2,",",p2,",",p2,") fit f: max resid = ",norm(fat2 .- ft,Inf))
Dj = [evalpoly(x,coq)^2-4*evalpoly(x,cor)*evalpoly(x,cop) for x in xj]
za,Dpza = find_near_roots(Dj, xj, rho=exp(1.0), meth="PR")
# the derivs Dpza all small at fake root pairs (doublets)
println("\tfirst root of D: ",za[1], ".  dist from true sing: ",abs(za[1]-z0))
if verb>0                    # plot error of f approximant in C-plane           
    fige=Figure();
    fag1 = [-evalpoly(z,cop)/evalpoly(z,coq) for z in zg]  # 2d grid eval -p/q
    axe,he=heatmap(fige[1,1],gx,gy,abs.(fag1.-fg), axis=(aspect=DataAspect(),))
    he.colorrange=10.0.^[-15,0]; he.colormap=:jet; he.colorscale=log10;
    scatter!(real.(xj),imag.(xj), label=L"x_j", color=:black, markersize=5)
    scatter!(real.(z0),imag.(z0), label="sing", marker='*')
    isnan(zz) || scatter!(real.(zz),imag.(zz), label="zero")
    Colorbar(fige[1,2],he, label="abs m=1 (rat) error")
    fag2 = [psi(z)[1] for z in zg]  # 2d grid eval psi, one branch
    axe,he=heatmap(fige[2,1],gx,gy,abs.(fag2.-fg), axis=(aspect=DataAspect(),))
    he.colorrange=10.0.^[-15,0]; he.colormap=:jet; he.colorscale=log10;
    scatter!(real.(xj),imag.(xj), label=L"x_j", color=:black, markersize=5)
    scatter!(real.(z0),imag.(z0), label="sing", marker='*')
    isnan(zz) || scatter!(real.(zz),imag.(zz), label="zero")
    Colorbar(fige[2,2],he, label="abs m=2 Hermite-Pade error")
    display(GLMakie.Screen(),fige)
end


