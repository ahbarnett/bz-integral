# test z2color
include("../src/z2color.jl")   # the thing to test

# basic tests, edge cases
for z in [1.0,1im,0.0,Inf,1im*Inf,NaN,1im*NaN]
    c = z2color(z)
    for cmpt in [red(c),green(c),blue(c)]
        if isnan(cmpt) || isinf(cmpt) error("z2color returned inf or nan")
        end
    end
end
println("z2color: edge cases tested")

# plot
using GLMakie
dx=0.05; L=10              # resolution, half-field-of-view
gx = range(-L,L,step=dx); gy = range(-L,L,step=dx)     # plot grid
zg = kron(gx,ones(size(gy'))) .+ 1im*kron(ones(size(gx)),gy')  # C grid
C = Makie.to_color(z2color.(zg))
fig,ax,h = heatmap(gx,gy,C, axis=(aspect=DataAspect(),))
GLMakie.activate!(title = "z2color: bare C-plane")
display(fig)
