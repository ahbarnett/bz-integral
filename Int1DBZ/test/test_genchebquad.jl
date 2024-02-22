# test script for generalized Chebyshev quadratures. Barnett 1/20/24
using Int1DBZ    # for miniquadgk
using Printf
using GLMakie

include("../src/genchebquad.jl")

function testquadfuncset(x::Vector, w::Vector, fs::Function,a,b,reqtol)
# make sure function set fs is integrated by rule (x,w) to requested tol
# returns the vector of errors of the rule for each function in the set.
    I,_,_,nev = miniquadgk(fs,a,b;rtol=1e-14,atol=1e-14);   # estim true i 
    Ig = reduce(hcat,fs.(x)) * w;    # test rule (x,w). reduce makes Nf*n mat (n=#nodes)
    maxerr = norm(I-Ig,Inf)
    @printf "max abs I err in fs vs miniquadgk(%d evals) = %.3g (vs tol=%.3g)\n" nev maxerr tol
    if maxerr<10tol; println("passed."); else println("failed!"); end
    I-Ig
end

tol=1e-12
for case = 0:3     # ---------------------------------- loop over func set choices

if case==0; println("Trivial test on monomials...")
    a=-1.0; b=1.0
    fs(x::Number) = [x^k for k=0:20]
elseif case==1; println("Basic test on non-integer power set...")
    a=0.0; b=1.0
    fs(x::Number) = [x^r for r=-0.53.+0.29*(0:30)]  #r=0.63*(-1:30)]     # set of irrational integrable powers >-1
    # *** note: fails for min r<-0.6. not sure why - try arb prec?
elseif case==2; println("poly plus nearby complex sqrt times poly...")
    a=-1.0; b=1.0
    p=20                  # max degree
    z0 = 0.3 + 1e-3im;    # sing loc near (a,b)
    #z0 = -1.001          # or, keeping it real, and sqrt away from its cut :)
    #fs(x::Number) = [[x^k for k=0:p]; [real(x^k/sqrt(x-z0)) for k=0:p];
    #    [imag(x^k/sqrt(x-z0)) for k=0:p]]        # separate Re, Im parts
    # neater way but ordering interleaved...    
    #fs(x::Number) = reduce(vcat, x^k.*[1, real(1/sqrt(x-z0)), imag(1/sqrt(x-z0))] for k=0:p)
    # even neater way using splat from tuple to Vector...
    fs(x::Number) = reduce(vcat, x^k.*[1, reim(1/sqrt(x-z0))...] for k=0:p)
elseif case==3; println("poly plus log|x-x0| times poly (x0 in interval)...")
    a=-1.0; b=1.0; p=20   # max degree
    x0 = 0.57;            # sing loc in (a,b)
    fs(x::Number) = reduce(vcat, x^k.*[1, log(abs(x-x0))] for k=0:p)
end

@time x, w, i = genchebquad(fs,a,b,tol;verb=1)
Ierrs = testquadfuncset(x,w,fs,a,b,tol)        # check the rule

# some test-cases outside the class... each time, fp must be deriv of f
if case!=1
    f(x) = sin(1+3x); fp(x) = 3cos(1+3x)
    @printf "analytic test x,w err: %.3g\n" abs(sum(w.*fp.(x))-f(b)+f(a))
end
if case==2
    #f(x) = sqrt(x-z0); fp(x) = 0.5/sqrt(x-z0)
    f(x) = sin(1+3x)*sqrt(x-z0); fp(x) = 0.5/sqrt(x-z0)*(sin(1+3x)+6(x-z0)*cos(1+3x))
    @printf "anal times 1/sqrt test x,w err: %.3g\n" abs(sum(w.*fp.(x))-f(b)+f(a))
end
if case==3
    #f(x) = (x-x0)*log(abs(x-x0)); fp(x) = 1 + log(abs(x-x0))   # too easy
    # note sinc is sin(pi.x)/(pi.x) in Julia...
    f(x) = sin(3*(x-x0))*log(abs(x-x0)); fp(x) = 3cos(3(x-x0))*log(abs(x-x0)) + 3sinc(3(x-x0)/pi)
    @printf "anal + log|x-x0|.anal test x,w err: %.3g\n" abs(sum(w.*fp.(x))-f(b)+f(a))
end

if true              # plot stuff from info struct i
    #GLMakie.closeall()
    GLMakie.activate!(title=@sprintf "test genchebquad case=%d" case);
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
    scatter!(abs.(Ierrs))
    lines!([0,Nf],[tol,tol],color=:red,label="tol")
    axislegend()
    #display(fig)
    display(GLMakie.Screen(),fig)       # new window
end

end                        # -------------------------------------------
