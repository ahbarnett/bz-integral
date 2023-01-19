# unit tester for contour 1D BZ module in this dir. Barnett 12/19/22

push!(LOAD_PATH,".")
using Cont1DBZ
using LinearAlgebra
using Printf
using OffsetArrays
using Test

# -------- module method tests ----------
M=10                 # 50 is high end
hm = OffsetVector(randn(ComplexF64,2M+1),-M:M)      # F-coeffs of h(x)
hm = (hm + conj(reverse(hm)))/2                     # make h(x) real

# test eval for x a scalar, vector, real, complex (each an el of tuple)...
nx = 1000
xtest = (1.9, [1.3], 2π*rand(nx), 2π*rand(nx)+im*rand(nx))
for t=1:length(xtest)
    @printf "evalh variants consistency: test #%d...\n" t
    local x = xtest[t]
    if t==1
        @printf "\tevalh @ x=%g: " x; println(evalh(hm,x))
    end
    @printf "evalh chk: %g\n" norm(evalh(hm,x) .- evalh_ref(hm,x),Inf)
    @printf "evalh_wind chk: %g\n" norm(evalh_wind(hm,x) .- evalh_ref(hm,x),Inf)
    @printf "fourier_kernel chk: %g\n" norm(fourier_kernel.(Ref(hm),x) .- evalh_ref(hm,x),Inf)
end

η=1e-6; ω=0.5; tol=1e-8;
@printf "test realadap for M=%d ω=%g η=%g tol=%g...\n" M ω η tol
Aa = realadap(hm,ω,η,tol=tol, verb=1)
    
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
    
# quadr: imag-shifted corrected PTR method...
NPTR=30
Ac = imshcorr(hm,ω,η,N=NPTR, verb=0)         # a=1 so Davis exp(-aN) ~ 1e-13
#println(Ac,' ',Aa)
@printf "test imshcorr, N_PTR=%d:   \t|Ac-Aa| = %g\n" NPTR abs(Ac-Aa)
# known band struc case with eta=0+
M=1; hm = OffsetVector(complex([1/2,0,1/2]),-M:M)    # h(x)=cos(x), band [-1,1]
ωs=[-1.6, -0.3, 0.9, 1.3, 1.9]          # below, thru, above band
η=0.0
for ω in ωs
    local Ac = imag(imshcorr(hm,ω,η,N=NPTR))   # Im for DOS
    Aanal = abs(ω)<1 ? 2π/sqrt(1-ω^2) : 0.0    # formula; DOS=0 outside band
    #println(Ac,' ',Aa)
    @printf "η=0+ Im test ω=%g:     \t|Ac-Aanal| = %g\n" ω abs(Ac-Aanal)
end

# ------------ end module tests---------------
