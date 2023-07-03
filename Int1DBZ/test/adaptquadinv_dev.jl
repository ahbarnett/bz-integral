# developing adaptquadinv - adaptive quadrature of inverse of analytic func
# via pole subtraction.
# Barnett, late June 2023.

using Int1DBZ
using Printf

# single-segment, single-pole, developing the math...
@printf "1-seg, f = 1/g, try pole subtract... vs 1e-10 tol numer quadr\n"
d = 1e-3
z0 = 0.3+1im*d
g(x) = 2*sin(x-z0)     # complex sin, root @ z0.  Next root dist ~1.8 from [a,b]
gp(x) = 2*cos(x-z0)    # g'
f(x::Number) = 1.0/g(x)
resf0 = 1.0/gp(z0)     # residue of f at its pole
a,b = -1.0,0.5
Im, Em, segs, numevals = miniquadgk(f,a,b,rtol=1e-10);  # right ans, slow
#plot(segs); @gp :- real(z0) imag(z0) "w p pt 1 ps 2 tit 'z_0'"
r = gkrule()
s = applygkrule(f,a,b,r)    # crude version w/ allocs
@printf "\tdumb uncorr 1-seg err %.3g (claimed E %.3g)\n" abs(s.I-Im) s.E
pole(x) = resf0./(x-z0)
sc = applygkrule(x->f(x)-pole(x), a,b,r)  # pole-sub func
Ic = sc.I + resf0*log((b-z0)/(a-z0))      # add exact pole integral
@printf "\tknown-pole corr 1-seg err %.3g (claimed E %.3g)\n" abs(Ic-Im) sc.E
# now fit roots & use resulting extracted residues...
mid, sca = (b+a)/2, (b-a)/2
xj = mid .+ sca*r.x
fj = f.(xj)       # our data on seg
ifj = 1.0./fj   # samples of analytic func
rs, ders = find_near_roots(ifj, r.x)
r0 = mid + sca*rs[1]; gp0 = ders[1]/sca;  # just treat 1st pole (there's only 1)
@printf "\tpole fit loc err %.3g, g'(root) err %.3g\n" abs(r0-z0) abs(gp0-gp(z0))
resfr = 1.0/gp0            # residue of f at its pole
polej = @. resfr/(xj-r0)   # pole vals at nodes  (maybe rewrite in std coords?)
sc = applygkrule(fj.-polej, a,b,r)       # pole-sub vals
Ic = sc.I + resfr*log((b-r0)/(a-r0))     # add exact pole integral
@printf "\tpole-fit corr 1-seg err %.3g (claimed E %.3g)\n" abs(Ic-Im) sc.E
# test self-contained func version...
sc = applypolesub!(ifj,fj,a,b,r)
@printf "\tapplypolesub 1-seg err %.3g (claimed E %.3g)\n" abs(sc.I-Im) sc.E
Ap,E,segs,numevals=adaptquadinv(g,a,b)
@printf "\tadaptquadinv err %.3g (%d segs, claimed E %.3g)\n" abs(Ap-Im) length(segs) E
@printf "\n"

