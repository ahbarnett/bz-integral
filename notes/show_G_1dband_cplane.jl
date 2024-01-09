# show H(x) = cos x Green's func in complex plane. Barnett 1/9/24

using GLMakie
include("../Int1DBZ/src/z2color.jl")

G(z::Number) = 1/(im*sqrt(1-z^2))    # 1d lattice Greens
dx=0.02; gx = range(-2,2,step=dx); gy = range(0,2,step=dx)  # plot grid
zg = kron(gx,ones(size(gy'))) .+ 1im*kron(ones(size(gx)),gy')  # C grid
Gg = G.(zg)
C = Makie.to_color(z2color.(Gg))
fig,ax,h=heatmap(gx,gy,C, axis=(aspect=DataAspect(),))
display(fig)
