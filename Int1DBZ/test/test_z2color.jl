# test z2color
using GLMakie
include("../src/z2color.jl")
dx=0.1; L=10
gx = range(-L,L,step=dx); gy = range(-L,L,step=dx)     # plot grid
zg = kron(gx,ones(size(gy'))) .+ 1im*kron(ones(size(gx)),gy')  # C grid
C = Makie.to_color(z2color.(zg))
fig,ax,h = heatmap(gx,gy,C, axis=(aspect=DataAspect(),))
GLMakie.activate!(title = "z2color: bare C-plane")
display(fig)
