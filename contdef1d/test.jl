# try contour def for 1d BZ integral, script
# Barnett 12/2/22

using QuadGK
using OffsetArrays
using BenchmarkTools

using Printf

# plotting
using Gnuplot
using ColorSchemes
Gnuplot.options.term="qt 0 font \"Sans,9\" size 1000,500"    # window pixels

# choose one energy band    h(x) = sum_{|m|<=M} hm exp(imx) Fourier series
M = 1   # max mod freq
hm = OffsetVector(complex([1/2,0,1/2]),-M:M)    # h(x)=cos(x)

"""
    evalh(hm,x)

    evaluate band Hamiltonian h(x) as complex Fourier series with coeffs hm
    (an offsetvector), possibly at vector of targets x
"""
function evalh(hm,x)
    N = length(x)
    Tout = eltype(complex(hm))
    h = zeros(Tout,size(x))
    for m in eachindex(hm)
        h = h .+ hm[m]*exp.(1im*m*x)
    end
    h
end
"""
    evalhp(hm,x)

    evaluate band Hamiltonian derivative h'(x) as complex Fourier series with
    coeffs hm (an offsetvector), possibly at vector of targets x
"""
function evalhp(hm,x)
    N = length(x)
    Tout = eltype(complex(hm))
    hp = zeros(Tout,size(x))
    for m in eachindex(hm)
        hp = hp .+ hm[m]*1im*m*exp.(1im*m*x)
    end
    hp
end
x=1.0; @printf "test evalh @ x=%g: " x; println(evalh(hm,x))
# varinfo()

η=1e-8      # broadening (quadgk obviously fails for small, below 1e-9)
ω=0.5       # Fermi energy
f(x) = 1 ./ (evalh(hm,x) .+ (-ω+1im*η))     # integrand func

ng = 1000    # Re axis plot grid
g=2pi*(0:ng-1)./ng
@gp :h g real(evalh(hm,g)) "w l t 'h(x) band'" xlab="Re x" xr=[0,2π]
@gp :h :- [0,2π] (ω-η)*[1,1] (ω+η)*[1,1] "w filledcu t 'omega +- eta' lc 'red' fs transparent solid 0.1"
fg = f(g)
@gp :f g imag(fg) "w l t 'Im'" tit="f(x) integrand" xlab="Re x" xr=[0,2π]
@gp :f :- g real(fg) "w l t 'Re'" "set xzeroaxis"

# ------------------------- Adaptive real quad method ----------------------

Aa,err = quadgk(f,0,2π,rtol=1e-8)            # A via adaptive real integral
@time Aa,err = quadgk(f,0,2π,rtol=1e-8)
println("A adap quadgk:\t",Aa)

# -------------------------- Deformed contour method ------------------------
"""
    fintegral, zj = contourPTR(f::Function,Cfun::Function,Cfunp::Function,N::Int)

    Apply N-node PTR to quadrature of f from 0 to 2π along contour
    Im z = Cfun(Re z). Cfunp must be derivative function of Cfun.
    To do: * remove need for Cfunp via spectral diff of Cfun on grid.
"""
function contourPTR(f::Function,Cfun::Function,Cfunp::Function,N::Int)
    tj=2π*(1:N)/N
    zj = tj .+ 1im*Cfun.(tj)
    wj = (1 .+ 1im*Cfunp.(tj)) * (2π/N)
    return sum(wj.*f.(zj)), zj
end
# Cfun, Cfunp = x->0.0.*x, x->0.0.*x; Ac,zj = contourPTR(f,Cfun,Cfunp,1000) # dumb Re test
Cfun, Cfunp = x->-sin(x), x->-cos(x)   # a contour good for |ω|<1
Ac,zj = contourPTR(f,Cfun,Cfunp,100)
Ac2,zj2 = contourPTR(f,Cfun,Cfunp,200)   # check
@printf "contour PTR diff:        \t%.3g\n" abs(Ac-Ac2)
@printf "contour PTR diff from Aa:\t%.3g\n" abs(Ac-Aa)

dx=0.03       # C plane plot grid
gx=range(0.0,2π,step=dx); gy=range(-1.0,1.0,step=dx)
gz = gx .+ 1im*gy'    # 2d array x-by-y (NB gnuplot flipped vs MATLAB)
fz = f(gz)            # func needs to preserve size
sc = 10               # color scale
@gp :C gx gy imag(fz) "w image notit" "set size ratio -1" palette(:jet1)
@gp :C :- cbr=[-sc,sc] xlab="Re x" ylab="Im x" tit="Im f(x)"
@gp :C :- "set autoscale fix"                                # tight
@gp :C :- real(zj) imag(zj) "w p pt 7 ps 0.3 lc '#000000' t 'cont PTR nodes'"


# --------------------- Residue corrected Im shift contour method ----------
using LinearAlgebra
"""
    roots(a)

    find all complex roots of polynomial a[1]*z^n + a[2]*z^(n-1) + ... + a[n+1]
    via companion matrix, in O(n^3) time.

    To do:
    * Bjoerck-Pereyra O(n^2) alg
"""
function roots(a::Vector)
    a = vec(a)              # make sure not funny shaped matrix
    a = complex(a)          # idempotent, unlike Complex{T} for T a type...
    T = eltype(a)
    while length(a)>1 && a[1]==0.0         # gobble up any zero leading coeffs
        a = a[2:end]
    end
    if isempty(a) || (a==[0.0])
        return NaN
    end
    deg = length(a)-1       # a is now length>1 with nonzero 1st entry
    if deg==0
        return T[]          # last is empty list of C-#s
    end
    a = reshape(a[deg+1:-1:2] ./ a[1],(deg,1))    # make monic, col and flip
    C = [ [zeros(T,1,deg-1); Matrix{T}(I,deg-1,deg-1)] -a ]   # stack companion mat
    E = eigen(C)    # but we don't want the vectors
    E.values
end
# println("roots test, should be +-im:"); roots([1.0,0,1.0])
# note a can't be a row-vec (matrix)
roots(complex([1.0,0,1.0]))       # complex case
roots([1.0])  # empty list
roots([0.0])  # all C-#s

hmconst = hm; hmconst[0] = hm[0] +  (-ω+1im*η)   # F series for denominator
zr = roots(hmconst[M:-1:-M])  # shift powers by M (must be std vec; reverse not)
xr = @. log(zr)/1im           # solve z = e^{ix}
xr = @. mod(real(xr),2π) + 1im*imag(xr)    # fold Re parts of roots into [0,2π)
println("poles in x plane: ", xr)
@gp :C :- real(xr) imag(xr) "w p pt 7 ps 1.0 lc '#ffffff' t 'poles'"

Cimsh = 1.0                 # contour imag const shift
Cfun, Cfunp = x->0*x.+Cimsh, x->0*x
Ar,zjr = contourPTR(f,Cfun,Cfunp,50)
resr = 1.0 ./ evalhp(hm,xr)     # residues of f at all poles ( = 1/h'(x_r))
for i in eachindex(xr)
    if imag(xr[i]) .> 0.0       # just the poles that contour passed through
        @printf "correcting res %g+%gi at pole %g+%gi...\n" real(resr[i]) imag(resr[i]) real(xr[i]) imag(xr[i])
        global Ar = Ar + 2π*1im*resr[i]   # correct by residue formula
    end
end
@printf "Res-corr PTR diff from Ac:\t%.3g\n" abs(Ar-Ac)
@gp :C :- real(zjr) imag(zjr) "w p pt 6 ps 0.3 lc '#000000' t 'Res-corr nodes'"
