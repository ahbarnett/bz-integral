# show Z^2 tight binding Green's func in complex plane. Barnett 1/10/24

using CairoMakie
include("../Int1DBZ/src/z2color.jl")
include("../Int1DBZ/src/ellipk.jl")

K(k) = ellipkAGM(k)          # local complete elliptic integral K
G2(z::Number) = 8pi*K(2/z)/z    # 2d lattice Greens vs omega
dx=0.03; gx = range(-3,3,step=dx); gy = range(0,2,step=dx)  # plot grid
zg = kron(gx,ones(size(gy'))) .+ 1im*kron(ones(size(gx)),gy')  # C grid
G2g = G2.(zg)
sc = 0.05
C = Makie.to_color(z2color.(sc*G2g))
fig,ax,h=heatmap(gx,gy,C, axis=(aspect=DataAspect(),))
ax.xlabel=L"Re $\omega$"; ax.ylabel=L"Im $\omega$"
display(fig)
save("2dband_cplane.png",fig)

L=3; gx = range(-L,L,step=0.01)
G2g = G2.(gx .+ 0*1e-15im)    # switch im part to check UHP lim correct 
fig,ax,h=lines(gx,real.(G2g),label=L"Re $G_2$");
lines!(gx,imag.(G2g),label=L"Im $G_2$",linestyle=:dash);
ax.xlabel=L"Re $\omega$"
sc = 50; ax.limits=(-L,L,-sc,sc)
axislegend()
display(fig)
save("2dband_real.png",fig)
