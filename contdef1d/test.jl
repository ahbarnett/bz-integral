# try contour def for 1d BZ integral, script. One omega, one eta, one N.
# Conpare 4 methods:
#   1) real quadGK
#   2) deformed contour PTR
#   3) residue-corr shifted PTR
#   4) cot-subtracted shifted PTR
# All work now; yet to test speeds
# Barnett 12/2/22.  Requirements: Julia (>=1.7 fine); gnuplot (>=5.4)

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
verb = 1

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
        h = h .+ hm[m]*exp.(im*m*x)
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
        hp = hp .+ hm[m]*im*m*exp.(im*m*x)
    end
    hp
end
x=1.0; @printf "test evalh @ x=%g: " x; println(evalh(hm,x))
# varinfo()

η=1e-1      # broadening (quadgk obviously hangs for small, below 1e-9)
ω=1.5       # Fermi energy (band is [-1,1]; 1.54 pole im~1.002 nr Cimsh=1)
#for ω = 0:0.03:2  # hacky loop
f(x) = 1 ./ (evalh(hm,x) .+ (-ω+im*η))     # integrand func

if verb>1
    ng = 1000    # Re axis plot grid
    g=2pi*(0:ng-1)./ng
    @gp :h g real(evalh(hm,g)) "w l t 'h(x) band'" xlab="Re x" xr=[0,2π]
    @gp :h :- [0,2π] (ω-η)*[1,1] (ω+η)*[1,1] "w filledcu t 'omega +- eta' lc 'red' fs transparent solid 0.1"
    fg = f(g)
    @gp :f g imag(fg) "w l t 'Im'" tit="f(x) integrand" xlab="Re x" xr=[0,2π]
    @gp :f :- g real(fg) "w l t 'Re'" "set xzeroaxis"
end


# ------------------------- Adaptive real quad method ----------------------

Aa,err = quadgk(f,0,2π,rtol=1e-8)            # A via adaptive real integral
@printf "timing quadgk... "
@time Aa,err = quadgk(f,0,2π,rtol=1e-8)
println("Aa adap quadgk:\t",Aa)              # for eta<1e-8, only good to 1e-7


# -------------------------- Deformed contour method ------------------------
"""
    fintegral, zj = contourPTR(f::Function,Cfun::Function,Cfunp::Function,N::Int)

    Apply N-node PTR to quadrature of f from 0 to 2π along contour
    Im z = Cfun(Re z). Cfunp must be derivative function of Cfun.
    Returns integral, and the complex nodes used zj.
    To do: * remove need for Cfunp via spectral diff of Cfun on grid.
"""
function contourPTR(f::Function,Cfun::Function,Cfunp::Function,N::Int)
    tj=2π*(1:N)/N
    zj = tj .+ im*Cfun.(tj)
    wj = (1 .+ im*Cfunp.(tj)) * (2π/N)
    return sum(wj.*f.(zj)), zj
end
# Cfun, Cfunp = x->0.0.*x, x->0.0.*x; Ac,zj = contourPTR(f,Cfun,Cfunp,1000) # dumb on-Re test
Cfun, Cfunp = x->-sin(x), x->-cos(x)   # a contour good for |ω|<1, dies -> 1.
# add more PTR nodes as poles coalesce...
    N=minimum([10000,maximum([50,ceil(Int,80/sqrt(abs(1-ω^2)))])])
    @printf "contour PTR N=%d\n" N
Ac,zj = contourPTR(f,Cfun,Cfunp,N)
Ac2,zj2 = contourPTR(f,Cfun,Cfunp,2N)   # self-conv
@printf "contour PTR self-conv:   \t%.3g\n" abs(Ac-Ac2)
@printf "contour PTR diff |Ac-Aa|:\t%.3g\n" abs(Ac-Aa)

if verb>0
    dx=0.03       # C plane plot grid
    gx=range(0.0,2π,step=dx); gy=range(-1.0,1.0,step=dx)
    gz = gx .+ im*gy'    # 2d array x-by-y (NB gnuplot flipped vs MATLAB)
    fz = f(gz)            # func needs to preserve size
    sc = 10               # color scale
    @gp :C gx gy imag(fz) "w image notit" "set size ratio -1" palette(:jet1)
    @gp :C :- cbr=[-sc,sc] xlab="Re x" ylab="Im x" tit="Im f(x)"
    @gp :C :- "set autoscale fix"                                # tight
    @gp :C :- real(zj) imag(zj) "w p pt 7 ps 0.3 lc '#000000' t 'cont PTR nodes'"
end


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

hmconst = copy(hm);           # crucial! otherwise changing hmconst changes hm!
hmconst[0] = hm[0] + (-ω+im*η)     # F series for denominator
zr = roots(hmconst[M:-1:-M])  # shift powers by M (must be std vec; reverse not)
xr = @. log(zr)/im            # solve z = e^{ix}
xr = @. mod(real(xr),2π) + im*imag(xr)   # fold Re parts into [0,2π), for humans
println("poles in x plane: ", xr)
@printf "largest denom mag at any pole (checks roots worked): %.3g\n" maximum(abs.(evalh(hm,xr) .+ (-ω+im*η)))
if verb>0
    @gp :C :- real(xr) imag(xr) "w p pt 7 ps 1.0 lc '#ffffff' t 'poles'"
end
    
Cimsh = 1.0                 # contour imag const shift
Cfun, Cfunp = x->0*x.+Cimsh, x->0*x
global Ar,zjr = contourPTR(f,Cfun,Cfunp,50)
@printf "Ar on shifted contour = %g+%gi\n" real(Ar) imag(Ar)
resr = 1.0 ./ evalhp(hm,xr)     # residues of f at all poles ( = 1/h'(x_r))
for i in eachindex(xr)
    if 0.0 < imag(xr[i]) < Cimsh   # just the poles that contour passed through
        @printf "correcting res %g+%gi at pole %g+%gi...\n" real(resr[i]) imag(resr[i]) real(xr[i]) imag(xr[i])
        global Ar = Ar + 2π*im*resr[i]   # correct by residue formula
    end
end
@printf "Res-corr PTR diff |Ar-Aa|: \t%.3g\n" abs(Ar-Aa)
#@printf "Res-corr PTR diff Im Ar-Aa:\t%.3g\n" imag(Ar-Aa)
@printf "Res-corr PTR diff |Ar-Ac|: \t%.3g\n" abs(Ar-Ac)
#@printf "Res-corr PTR diff Im Ar-Ac:\t%.3g\n" imag(Ar-Ac)
if verb>0
    @gp :C :- real(zjr) imag(zjr) "w p pt 6 ps 0.7 lc '#000000' t 'Res-corr nodes'"
end


# --------------------- cot-subtraction Im shift contour method --------------
# * kill Im(pole) Cimsh-near PTR contour's loss of accuracy by cot((x-xr)/2)
#   subtraction (using analytic form of cot C-integral: ±2π dep on Im xr).
# * reuses the Cimsh, Cfun{p}, poles xj & resr from above
N = 50     # start with uncorr shifted PTR (quadr scheme must stay fixed)...
global As,zjr = contourPTR(f,Cfun,Cfunp,N)
for i in eachindex(xr)
    if 0.0 < imag(xr[i]) < Cimsh   # pole crossed, near im-shifted PTR contour
        @printf "cot-subtracting res %g+%gi at below-cont res-corr pole %g+%gi...\n" real(resr[i]) imag(resr[i]) real(xr[i]) imag(xr[i])
        global Acot,zjr = contourPTR(x -> cot((x-xr[i])/2),Cfun,Cfunp,N)
        global As = As - resr[i]/2*(Acot - 2π*im)     # cot and the Res thm add
    elseif Cimsh < imag(xr[i]) < 2Cimsh   # pole not crossed, still near cont
        @printf "cot-subtracting res %g+%gi at above-cont pole %g+%gi...\n" real(resr[i]) imag(resr[i]) real(xr[i]) imag(xr[i])
        global Acot,zjr = contourPTR(x -> cot((x-xr[i])/2),Cfun,Cfunp,N)
        global As = As - resr[i]/2*(Acot - 2π*im)     # sub PTR on cot, add ex
    end
end
# *** understand and combine above cases!
@printf "cot-sub PTR diff |As-Aa|: \t%.3g\n" abs(As-Aa)
@printf "cot-sub PTR diff |As-Ac|: \t%.3g\n" abs(As-Ac)
# *** todo replot new integrand in C. check smooth on contour PTR nodes too
    
