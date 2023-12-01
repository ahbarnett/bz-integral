# try Hermite-Pade on std interval for fitting rational or
# inv-sqrt singular functions.  Barnett 12/1/23
using Int1DBZ
using GLMakie
verb=1
d = 1e-1         # imag dist
z0 = 0.3+1im*d    # sing loc
f(x::Number) = 1.0 + (2-1im)/sin(x-z0))     # sqrt sing @ z0, branch cut not hit interval. Next root dist ~1.8 from [a,b]
