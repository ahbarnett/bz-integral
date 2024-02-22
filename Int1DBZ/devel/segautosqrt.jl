# try combining quadratic pade with GCQ for auto-sqrt handling on 1 seg
# Barnett 2/22/24

G1(om) = 2pi/(1im*sqrt(1-om^2))       # x-integral done in tight-binding model
f(y) = G1(om + 1im*eta - cos(y))     # integrand for middle integral

