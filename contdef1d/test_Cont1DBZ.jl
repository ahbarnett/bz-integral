# unit tester for contour 1D BZ module in this dir.
# Barnett 12/19/22. LXVM added fourier_kernel which is an evalh replacement.
# Started matrix (n>1) case 1/20/23.

push!(LOAD_PATH,".")
using Cont1DBZ
using LinearAlgebra
using Printf
using OffsetArrays
using Test
using StaticArrays
using QuadGK

# -------- module method tests ----------
M=10         # max mag Fourier freq index
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # F-coeffs of h(x)
hm = (hm + conj(reverse(hm)))/2                     # make h(x) real for x Re

n = 3        # vector (tight-binding) case
# pure OA versions.
#Hm = OffsetArray(randn(ComplexF64,(n,n,2M+1)),1:n,1:n,-M:M)  # n*n*(2M+1)
#Hmconj = permutedims(conj(Hm),(2,1,3))    # hermitian transpose wrt dims (1,2)
#Hm = (Hm + (reverse(Hmconj,dims=3)))/2              # H(x) hermitian if x Re
mlist = -M:M   # OV of SAs version
Hm = OffsetVector([SMatrix{n,n}(randn(ComplexF64,(n,n))) for m in mlist], mlist)
Hmconj = OffsetVector([Hm[m]' for m in mlist], mlist)   # ugh! has to be better!
Hm = (Hm + reverse(Hmconj))/2                           # H(x) hermitian if x Re

# test eval for x a scalar, vector, real, complex (each an el of tuple)...
nx = 1000
xtest = (1.9, [1.3], 2π*rand(nx), 2π*rand(ComplexF64, nx))
for (t,x) in enumerate(xtest)
    @printf "scalar evalh variants consistency: test #%dS...\n" t
    if t==1
        @printf "\tevalh @ x=%g: " x; println(evalh(hm,x))
    end
    @printf "evalh chk:          %.3g\n" norm(evalh(hm,x) - evalh_ref(hm,x),Inf)
    @printf "evalh_wind chk:     %.3g\n" norm(evalh_wind(hm,x) - evalh_ref(hm,x),Inf)
    @printf "fourier_kernel chk: %.3g\n" norm(fourier_kernel.(Ref(hm),x) - evalh_ref(hm,x),Inf)
    @printf "matrix evalh variants consistency: test #%dM...\n" t
    H = evalh_ref(Hm,x)
    if t<=2
        @printf "evalh_ref simply check is Herm: %.3g\n" norm(H - H',Inf)
    end
    @printf "evalh chk:          %.3g\n" norm(evalh(Hm,x) - evalh_ref(Hm,x),Inf)
    @printf "evalh_wind chk:     %.3g\n" norm(evalh_wind(Hm,x) - evalh_ref(Hm,x),Inf)
    @printf "fourier_kernel chk: %.3g\n" norm(fourier_kernel.(Ref(Hm),x) - evalh_ref(Hm,x),Inf)
end

η=1e-6; ω=0.5; tol=1e-8;
@printf "\nConventional quadrature (eta>0, obvi):\n"
@printf "test realadap for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Aa = realadap(hm,ω,η,tol=tol, verb=1)
@printf "\tAa = "; println(Aa)
@printf "test realadap, n=%d (matrix), M=%d ω=%g η=%g tol=%g...\n" n M ω η tol
Aam = realadap(Hm,ω,η,tol=tol, verb=1)       # suffix m for matrix
@printf "\tAam = "; println(Aam)


@printf "\nTest roots methods (also see bench_roots.jl)...\n"
@testset "roots" begin
r = roots([1.0,0,1.0])                # real coeffs case
@test maximum(abs.(r).-1.0) < 1e-14 && (sort(angle.(r)) ≈ [-pi/2,pi/2])  # +-i
r = roots(complex([1.0,0,1.0]))       # complex case
@test maximum(abs.(r).-1.0) < 1e-14 && (sort(angle.(r)) ≈ [-pi/2,pi/2])  # +-i
# dumb cases
@test roots([1.0]) == []              # triv case no roots
r = roots([0.0])                      # all C-#s
@test isnan(r[1]) && (length(r)==1)
end
@testset "roots_best" begin
r = roots_best([1.0,0,1.0])                # real coeffs case
@test maximum(abs.(r).-1.0) < 1e-14 && (sort(angle.(r)) ≈ [-pi/2,pi/2])  # +-i
r = roots_best(complex([1.0,0,1.0]))       # complex case
@test maximum(abs.(r).-1.0) < 1e-14 && (sort(angle.(r)) ≈ [-pi/2,pi/2])  # +-i
# dumb cases (may differ from above roots())
@test roots_best([1.0]) == []              # triv case no roots
#r = roots_best([0.0])                      # seems like can't handle
end

@printf "\nNew quadrature methods, scalar case... (for above params)\n"
# imag-shifted quadr corrected PTR quadr method...
NPTR=30
Ac = imshcorr(hm,ω,η,N=NPTR, verb=0)         # a=1 so Davis exp(-aN) ~ 1e-13
#println(Ac," ",Aa)
@printf "test imshcorr, N_PTR=%d:   \t|Ac-Aa| = %.3g    (don't trust below claimed err)\n" NPTR abs(Ac-Aa)

# disc residue thm method...
Ad = discresi(hm,ω,η,verb=0)
#println(Ad,"    ",Aa,"   ratio:",Ad/Aa)   # for eyeball ratio-fixing :)
@printf "test discresi:\t\t\t|Ad-Aa| = %.3g    (don't trust below claimed err)\n" abs(Ad-Aa)

@printf "\nNew quadrature methods, matrix case... (for above params)\n"
Adm = discresi(Hm,ω,η, verb=1)
println(Adm,"    ",Aam,"   ratio:",Adm/Aam)   # for eyeball ratio-fixing :)
@printf "test discresi:\t\t\t|Adm-Aam| = %.3g    (don't trust below claimed err)\n" abs(Adm-Aam)


# known band struc case with eta=0+: imshcorr...
@printf "\nNew quadrature methods at eta=0+, scalar case...\n"
M=1; hm = OffsetVector(complex([1/2,0,1/2]),-M:M)    # h(x)=cos(x), band [-1,1]
# let's see if can break via zero extreme coeffs... (now fixed)
#M=2; hm = OffsetVector(complex([0,1/2,0,1/2,0]),-M:M)    # h(x)=cos(x), band [-1,1]
η=0.0   # poss to check eta>0 against below exact DOS form w O(eta) error too
ωs=[-1.6, -0.3, 0.9, 1.3, 1.9]          # below, thru, above band
for ω in ωs
    DOS = abs(ω)<1 ? 2/sqrt(1-ω^2) : 0.0    # exact form; DOS=0 outside band
    local DOSc = (-1/π)*imag(imshcorr(hm,ω,η,N=NPTR))   # -Im(A)/pi for DOS
    local DOSd = (-1/π)*imag(discresi(hm,ω,η, verb=0))
#    println("DOS=",DOS,", DOSc=",DOSc,", DOSd=",DOSd)  # debug
    @printf "η=0+ Im test ω=%g:     \tDOSc err = %.3g          \tDOSd err = %.3g\n" ω DOSc-DOS DOSd-DOS
end

# graphene band structure and DOS, with coefficient definition explained below
# https://lxvm.github.io/AutoBZ.jl/dev/pages/demo/#DOS-of-Graphene
hm = OffsetMatrix(zeros(SMatrix{2,2,ComplexF64,4}, (5,5)), -2:2, -2:2)
hm[1,1]   = hm[1,-2] = hm[-2,1] = [0 1; 0 0]
hm[-1,-1] = hm[-1,2] = hm[2,-1] = [0 0; 1 0]

ω = 2.0; η = 0.1

# gDOSc_int = DOSIntegral1D(imshcorr, (ω, η), (;), hm)
# gDOSc = quadgk(gDOSc_int, 0, 2pi; rtol=1e-5) # ERROR: MethodError: no method matching imshcorr ... matrix-valued coefficients

# gDOSd_int = DOSIntegral1D(discresi, (ω, η), (;), hm)
# gDOSd = quadgk(gDOSd_int, 0, 2pi; rtol=1e-5) # ERROR: ArgumentError: invalid argument #4 to LAPACK call

gDOSa_int = DOSIntegral1D(realadap, (ω, η), (; tol=1e-5, kernel=fourier_kernel), hm)
gDOSa = quadgk(gDOSa_int, 0, 2pi; rtol=1e-5)
# probably need to multiply by Jacobian determinant

# ------------ end module tests---------------
