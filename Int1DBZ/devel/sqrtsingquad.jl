# custom quadr for p(z) + q(z)/sqrt(z-z0) where p,q polynomials.
# Barnett 1/11/24
using Int1DBZ
using FastGaussQuadrature
using LinearAlgebra

# Vandermonde-transpose solve on [-1,1]
d = 1e-2         # dist
x0 = 0.3           # x-loc of sing
z0 = x0+1im*d   # sing loc, Complex type

#xj,~ = gausslegendre(16)   # kappa(V) and ||v|| stabilize as N grows
r = gkrule(); xj = r.x
#xj = [xj; x0]     # add a near node
N = length(xj)
xj = Complex.(xj)       # so V entries in C

dt = 8;  # number of functions to try to integrate
dq = dt÷2-1; dp = dt-dq-2    # p,q poly degrees (allows p>q)
Vp = [x^k for x in xj, k in 0:dp]
Vq = [x^k./sqrt(x-z0) for x in xj, k in 0:dq]
VT = [Vp Vq]'
println("kappa(V)=",cond(VT))
bp = [mod(k,2)==0 ? 2/(k+1) : 0 for k in 0:dp]   # exact monomial integrals
bq = zeros(eltype(xj),dq+1)         # store exact sqrt.monomial integrals...
bq[1] = 2(sqrt(1-z0)-sqrt(-1-z0))   # k=0, analytic
for k=1:dq          # recur upwards, I_k stored in bq[k+1]
    bq[k+1] = (sqrt(1-z0)-(-1)^k*sqrt(-1-z0) + k*z0*bq[k])/(0.5+k)
end
if false         # test recurrence worked vs numer integral
    for k=0:dq 
        Ik,~ = miniquadgk(z -> z^k/sqrt(z-z0), -1.0, 1.0, rtol=1e-14)
        println(k,"\t",bq[k+1]-Ik)
    end
end
rhs = [bp;bq]
vj = VT \ rhs
println("resid 2-nrm: ",norm(VT*vj-rhs),"\t||v||_1=",norm(vj,1))
# 1e8; we lose 8 digits of inner integral acc; bad!




