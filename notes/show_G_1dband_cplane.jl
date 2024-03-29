# show H(x) = cos x Green's func in complex plane. Barnett 1/9/24

using CairoMakie #LaTeXStrings
include("../Int1DBZ/src/z2color.jl")

G(z::Number) = 1/(im*sqrt(1-Complex(z)^2))    # 1d lattice Greens
dx=0.02; gx = range(-2,2,step=dx); gy = range(0,2,step=dx)  # plot grid
zg = kron(gx,ones(size(gy'))) .+ 1im*kron(ones(size(gx)),gy')  # C grid
Gg = G.(zg)
C = Makie.to_color(z2color.(Gg))
fig,ax,h=heatmap(gx,gy,C, axis=(aspect=DataAspect(),))
ax.xlabel=L"Re $\omega$"; ax.ylabel=L"Im $\omega$"
display(fig)
save("1dband_cplane.png",fig)

gx = range(-2,2,step=0.005)
Gg = G.(gx .+ 0*1e-12im)    # switch on the im part to check the UHP lim
fig,ax,h=lines(gx,real.(Gg),label=L"Re $G_1$");
lines!(gx,imag.(Gg),label=L"Im $G_1$",linestyle=:dash);
ax.xlabel=L"Re $\omega$"
ax.limits=(-2,2,-5,5)
axislegend()
display(fig)
save("1dband_real.png",fig)
