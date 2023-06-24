####### Conventional real-axis quadrature methods...

"""
    A = realadap(hm,ω,η;tol,verb)

    use quadgk on Re axis to integrate 1/(ω - h(x) + iη).
    hm is given by offsetvector of Fourier series. tol controls rtol.
"""
function realadap(hm,ω,η; tol=1e-8, verb=0, kernel=evalh_ref)
    f(x::Number) = tr(inv(complex(ω,η)*I - kernel(hm,x)))    # integrand func (quadgk gives x a number)
    A,err = quadgk(f,0,2π,rtol=tol)          # can't get more info? # fevals?
    if verb>0
        @printf "\trealadap claimed err=%g\n" err
    end
    A
end

"""
    A = realadap_lxvm(hm,ω,η;tol,verb)

    use quadgk on Re axis to integrate 1/(ω - h(x) + iη), via
    faster non-allocating 1D Fourier series evaluator.
    hm is given by offsetvector of Fourier series. tol controls rtol.
    By LXVM.
"""
realadap_lxvm(hm, ω, η; tol=1e-8, verb=0) = realadap(hm, ω, η; tol=tol, verb=verb, kernel=fourier_kernel)

