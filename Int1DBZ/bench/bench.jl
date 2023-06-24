using Int1DBZ
using Printf

η=1e-6; ω=0.5; tol=1e-8;
@printf "\nConventional quadrature (eta>0, obvi):\n"
@printf "test realadap for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Aa = realadap(hm,ω,η,tol=tol, verb=1)
