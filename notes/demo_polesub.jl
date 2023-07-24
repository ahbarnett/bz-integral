# simple demo pole-sub on [a,b], with single Gauss rule.
# Figs for adaptoplesub.tex
# Barnett 7/24/23
using Int1DBZ
using FastGaussQuadrature

a=-1.0;b=1.0;
p=16    # rule we used. interestingly pole-sub dies badly for p<10
xj,wj=gausslegendre(p)

d = 1e-3   # test func: dist of pole
z0 = 0.3+1im*d       # pole, not near node (dist >0.018)
#z0 = xj[10] + 0*0.7*d + 1im*d    # near a node "pole-hitting" failure
func = "ord2pole"
verb = 0       # 1 overwrites .eps
if func=="invsin"
    om = 2      # controls density of poles
    f(x) = 1.0/sin(om*(x-z0))        # sin has roots @ z0 + integer*pi/om
    F(x) = log(1.0/sin(om*(x-z0)) - cot(om*(x-z0))) / om    # antideriv
elseif func=="cot"
    om = 0.9      # controls density of poles and zeros
    f(x) = cot(om*(x-z0))        # poles @ z0 + integer*pi/om, zeros inbetween
    F(x) = log(-sin(om*(x-z0))) / om   # note branch cut rot via minus sign
    println("some zeros: ", z0.+π/om*(-0.5.+collect(-2:2)))
elseif func=="ord2pole"
    om = 2      # controls density of poles and zeros
    f(x) = 1.0/sin(om*(x-z0))^2  # double-poles @ z0 + integer*pi/om
    F(x) = -cot(om*(x-z0)) / om
end
I0 = F(b)-F(a)
println("analytic I=",I0)
Igk, Egk, segs, numevals = miniquadgk(f,a,b,rtol=1e-12);
println("Igk=       ",Igk,":\n\tEgk=",Egk," fevals=",numevals,"\terr=",abs(Igk-I0))

println("min dist pole from node = ",minimum(abs.(xj.-z0)))
fj = f.(xj)          # create data
rho = 1.0            # Bern param
Ipoles = 0.0        # do pole-sub...
roots, gproots = find_near_roots( 1.0./fj, xj, rho=rho,meth="PR")  # steps 1+2
if verb>0
    using CairoMakie, LaTeXStrings, Colors   # or using GLMakie for interactive
    fig,ax, = scatter(real(roots),imag(roots),label=L"roots $r_k$");
    scatter!(ax,xj,0*xj, marker='+',label=L"nodes $x_j$");
    ax.xlabel=L"Re $x$"; ax.ylabel=L"Im $x$"
    if func=="cot"; rf=z0-π/om/2; scatter!(real(rf),imag(rf),marker='x',label=L"roots of $f$"); end
    z=exp.(rho .+ 1im*range(0,2π,100)); t=@.(z+1.0/z)/2;
    lines!(real(t),imag(t),label=L"$\rho=1$ Bernstein ellipse",color=RGB(0.5,0,0))   # RGB() needs using Colors explicitly
    axislegend(); ax.xgridvisible=ax.ygridvisible=false
    resize!(fig,(600,400))         # seems to scale from default (800,600)
    display(fig)
    save("../notes/spurious.eps",fig)
    # or...
    #using Gnuplot
    #@gp real(roots) imag(roots) "w p pt 1" xj 0*xj "w p pt 2"
    #@gp :- "set size ratio -1"
end
for (i,r) in enumerate(roots)
    Res = 1.0/gproots[i]    # step 3
    println("\tpole #",i,": r=",r,",  1/|f(r)|=",1.0/abs(f(r)),"  Res=",Res)
    @. fj -= Res/(xj-r)       # subtract, step 4
    global Ipoles += Res * log((b-r)/(a-r))
end
I = sum(wj.*fj) + Ipoles      # integrate corrected f_j vals, plus analytic
println("I=",I,": err=",abs(I-I0))
