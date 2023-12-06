# plot roots z+ and z- for G(om) derivation in d=1
using GLMakie
function zr(om::Number)        # pair of roots via quadratic formula
    s = sqrt(1-om^2)          # negate discriminant to use usual sqrt
    [om+im*s; om-im*s]
end
eta=0.1; om=range(-2+eta*im,2+eta*im,length=100)
r = stack(zr.(om))
fig,ax,h = scatter(real.(r[1,:]),imag.(r[1,:]),marker='o',color=real.(om),label="z+")
scatter!(real.(r[2,:]),imag.(r[2,:]),marker='x',color=real.(om),label="z-")
t=range(0,2pi,length=100); lines!(cos.(t), sin.(t))
h.colormap=:jet; Colorbar(fig[1,2],h,label=L"Re $\omega$")
ax.aspect=DataAspect();
axislegend()
display(fig)