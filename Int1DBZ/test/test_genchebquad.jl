# test script for generalized Chebyshev quadratures
using Int1DBZ    # for miniquadgk
using Printf
using GLMakie

include("../src/genchebquad.jl")

function testquadfuncset(x::Vector, w::Vector, fs::Function,a,b,reqtol)
# make sure function set fs is integrated by rule (x,w) to requested tol
    I,_,_,nev = miniquadgk(fs,a,b;rtol=1e-14,atol=1e-14);   # estim true i 
    Ig = reduce(hcat,fs.(x)) * w;    # test rule (x,w). reduce makes Nf*n mat (n=#nodes)
    maxerr = norm(I-Ig,Inf)
    @printf "max abs I err in fs vs miniquadgk(%d evals) = %.3g (vs tol=%.3g)\n" nev maxerr tol
    if maxerr<10tol; println("passed."); else println("failed!"); end
    maxerr
end

tol=1e-12

case = 0
if case==0; println("Trivial test on monomials...")
    a=-1.0; b=1.0
    fs(x::Number) = [x^k for k=0:20]
elseif case==1; println("Basic test on non-integer power set...")
    a=0.0; b=1.0
    fs(x::Number) = [x^r for r=-0.53.+0.29*(0:30)]  #r=0.63*(-1:30)]     # set of irrational integrable powers >-1
    # *** note: fails for r>0.6 :(
elseif case==2; println("poly plus nearby complex sqrt times poly...")
    a=-1.0; b=1.0
    p=20            # max degree
    z0 = 0.3 + 1e-3im;    # sing loc near (a,b)
    #z0 = -1.001   # or, keeping it real, and sqrt away from its cut :)
    #fs(x::Number) = [[x^k for k=0:p]; [real(x^k/sqrt(x-z0)) for k=0:p];
    #    [imag(x^k/sqrt(x-z0)) for k=0:p]]        # separate Re, Im parts
    # neater way but ordering interleaved...    
    #fs(x::Number) = reduce(vcat, x^k.*[1, real(1/sqrt(x-z0)), imag(1/sqrt(x-z0))] for k=0:p)
    # even neater way using splat from tuple to Vector...
    fs(x::Number) = reduce(vcat, x^k.*[1, reim(1/sqrt(x-z0))...] for k=0:p)
end

x, w, i = genchebquad(fs,a,b,tol;verb=1)
testquadfuncset(x,w,fs,a,b,tol)        # check the rule

if case!=1
    f(x) = sin(1+3x); fp(x) = 3*cos(1+3x)   # fp must be deriv of f
    @printf "analytic test x,w err: %.3g\n" abs(sum(w.*fp.(x))-f(b)+f(a))
end
if case==2
    f(x) = sqrt(x-z0); fp(x) = 0.5/sqrt(x-z0)
    @printf "anal times 1/sqrt test x,w err: %.3g\n" abs(sum(w.*fp.(x))-f(b)+f(a))
end

if true              # plot stuff from info struct i
    GLMakie.closeall()
    GLMakie.activate!(title="test genchebquad");
    fig=Figure(); ax1=Axis(fig[1,1],title="input func set")
    Nf = length(fs(a));   # num input funcs
    t=range(a,b,1000); F=reduce(hcat,fs.(t));  # eval all fs to plot
    for fj in eachrow(F); lines!(t,real.(fj)); end
    ax2=Axis(fig[1,2],title="u o.n. funcs at dense nodes")
    m,r=size(i.U)
    for j=1:r             # cancel out sqrt-w factors to view raw u funcs
        scatterlines!(i.xg,real.(i.U[:,j])./sqrt.(i.wg),markersize=5)
    end
    ax3=Axis(fig[2,1],title="rule: w_j at each x_j")
    scatter!(x,w)
    linesegments!(kron(x,[1;1]),kron(w,[0;1])) # stick plot
    ax4=Axis(fig[2,2], yscale=log10, limits=(nothing,(1e-16,1)),
        title="abs I err over func set")
    scatter!(abs.(I-Ig))
    lines!([0,Nf],[tol,tol],color=:red)
    display(fig)
    #display(GLMakie.Screen(),fig)       # new window
end
