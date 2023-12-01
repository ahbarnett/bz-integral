# try Hermite-Pade on std interval for fitting rational or
# inv-sqrt singular functions.  Based on fit_invsqrt.jl
# Barnett 12/1/23
using Int1DBZ
using GLMakie
using LinearAlgebra
verb=1

d = 1e-2         # imag dist of sing
println("try_hermitepade...     sing dist d=",d)
z0 = 0.3+1im*d    # sing loc
zz = -0.5-0.2im    # nearby zero loc
f(x::Number) = sin(x-zz)/sin(x-z0)    # zero & pole, not a rat
#f(x::Number) = (x-zz)/(x-z0)          # (1,1) rat

r = gkrule()     # GK rule on [-1,1]
xj = r.x        # the larger node set
N = length(xj)
fj = f.(xj)    # data
t = range(-1.0,1.0,1000)     # for cont curve plot & max estim on [-1,1]
ft = f.(t)                   # true vals on dense grid on [-1,1]
if verb>0
    fig=Figure();           # two-panel plot of vals on interval, C-plane
    ax,=scatter(fig[1,1],xj,real.(fj), label=L"Re $f_j$")       # nodes
    scatter!(xj,imag.(fj), label=L"Im $f_j$")
    lines!(t,real.(ft))           # curve
    lines!(t,imag.(ft))
    ax.xlabel=L"$t$"
    axislegend()
    dx=0.02
    gx = range(-2,2,step=dx); gy = range(-1,1,step=dx)
    zg = kron(gx,ones(size(gy'))) .+ 1im*kron(ones(size(gx)),gy')
    fg = f.(zg)      # eval
    ax2,h=heatmap(fig[2,1],gx,gy,abs.(fg), axis=(aspect=DataAspect(),))
    h.colorrange=10.0.^[-2,2]; h.colormap=:jet; h.colorscale=log10;
    Colorbar(fig[2,2],h, label=L"|f|")
    scatter!(real.(xj),imag.(xj), label=L"x_j", color=:black, markersize=5)
    scatter!(real.(z0),imag.(z0), label="sing", marker='*')
    scatter!(real.(zz),imag.(zz), label="zero")
    axislegend()
    display(fig)
end
r,gpr = find_near_roots(1.0./fj, xj, rho=exp(1.0), meth="PR")
println("\tpole loc err in find_near_roots on 1/f = ",abs(r[1]-z0))
p = N-1      # degree
V = [x^i for x in xj, i in 0:p]    # Vandermonde
co = V\fj
fat = [evalpoly(x,co) for x in t]          # f's approximant on t grid
println("\tpoly(",p,") fit f: max resid = ",norm(fat .- ft,Inf))
p1 = 6      # degree of p and q. Do Gonnet-G\uttelTrefethen fit via SVD
V1 = V[:,1:p1+1]; V1 = [V1 diagm(fj)*V1]
S = svd(V1)
println("sing vals: ", S.S)
nullco = S.Vt[end,:]'              # ' = conj transp, needed!
cop = nullco[1:p1+1]; coq = nullco[p1+2:end]    # extract p,q poly coeffs
fat1 = [-evalpoly(x,cop)/evalpoly(x,coq) for x in t]  # grid eval rat -p/q
println("\trat(",p1,",",p1,") fit f: max resid = ",norm(fat1 .- ft,Inf))
if verb>0
    fige=Figure();           # error of f in C-plane
    fag1 = [-evalpoly(z,cop)/evalpoly(z,coq) for z in zg]  # 2d grid eval -p/q
    axe,he=heatmap(fige[1,1],gx,gy,abs.(fag1.-fg), axis=(aspect=DataAspect(),))
    he.colorrange=10.0.^[-15,0]; he.colormap=:jet; he.colorscale=log10;
    scatter!(real.(xj),imag.(xj), label=L"x_j", color=:black, markersize=5)
    scatter!(real.(z0),imag.(z0), label="sing", marker='*')
    scatter!(real.(zz),imag.(zz), label="zero")
    Colorbar(fige[1,2],he, label="abs error")
    display(GLMakie.Screen(),fige)
end


