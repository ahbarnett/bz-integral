####### Conventional real-axis quadrature methods using QuadGK directly...

"""
    A = realadap(hm,ω,η;tol,verb)

    use quadgk on Re axis to integrate 1/(ω - h(x) + iη).
    hm is given by offsetvector of Fourier series. tol controls rtol.
"""
function realadap(hm,ω,η; tol=1e-8, verb=0, kernel=evalh_ref)
    f(x::Number) = tr(inv(complex(ω,η)*I - kernel(hm,x)))    # integrand func (quadgk gives x a number; note I is Id if StaticArray matrix)
    if verb>0
        A,err,fevals = quadgk_count(f,0,2π,rtol=tol)
        @printf "\trealadap: fevals=%d,  claimed err=%g\n" fevals err
    else
        A,err = quadgk(f,0,2π,rtol=tol)
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


"""
    A = realmyadap(hm,ω,η;tol)

    use miniquadgk on Re axis to integrate 1/(ω - h(x) + iη).
    hm is given by offsetvector of Fourier series. tol controls rtol.
    Scalar `h(x)` only
"""
function realmyadap(hm,ω,η; tol=1e-8)
    f(x::Number) = inv(complex(ω,η) - fourier_kernel(hm,x))
    return miniquadgk(f,0.0,2π,rtol=tol)
end



########### New methods...
