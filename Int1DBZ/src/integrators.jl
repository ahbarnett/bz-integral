####### Conventional real-axis quadrature methods using QuadGK directly...

"""
    A = realadap(hm,ω,η;tol,verb)

    use quadgk on Re axis to integrate 1/(ω - h(x) + iη).
    `hm` is given by offsetvector of Fourier series, either scalar- or
    matrix-valued. `tol` controls `rtol`.
"""
function realadap(hm,ω,η; tol=1e-8, verb=0, kernel=evalh_ref)
    # integrand (quadgk gives x a number; note I is Id if StaticArray matrix)
    f(x::Number) = tr(inv(complex(ω,η)*I - kernel(hm,x)))
    # note how `kernel` lets the evaluator be general
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

    use quadgk on Re axis to integrate 1/(ω - h(x) + iη), via faster
    non-allocating 1D Fourier series evaluator.  `hm` is given by
    offsetvector of Fourier series, either scalar- or
    matrix-values. `tol` controls `rtol`.  By LXVM.
"""
realadap_lxvm(hm, ω, η; tol=1e-8, verb=0) =
    realadap(hm, ω, η; tol=tol, verb=verb, kernel=fourier_kernel)


"""
    A = realmyadap(hm,ω,η;tol,ab)

    use miniquadgk on Re axis to integrate 1/(ω - h(x) + iη).
    hm is given by offsetvector of Fourier series, either scalar- or
    matrix-values.
    `tol` controls `rtol`.
    `ab` allows custom domain, default [0,2π].
    Uses AHB's miniquadgk, hinted by `my` in function name.
"""
function realmyadap(hm,ω,η; tol=1e-8, ab=[0.0,2π])
    # as in realadap, tr is Trace, has no effect in scalar, inv is reciprocal
    f(x::Number) = tr(inv(complex(ω,η)*I - fourier_kernel(hm,x)))
    return miniquadgk(f,ab[1],ab[2],rtol=tol)
end


########### New methods...

"""
    A = realquadinv(hm,ω,η;tol)

    use adaptquadinv on Re axis to integrate 1/(ω - h(x) + iη).
    hm is given by offsetvector of Fourier series. tol controls rtol.
    Scalar `h(x)` only for now.
"""
function realquadinv(hm,ω,η; tol=1e-8, rho=1.0, verb=0, ab=[0.0,2π])
    g(x::Number) = complex(ω,η) - fourier_kernel(hm,x)
    return adaptquadinv(g,ab[1],ab[2],rtol=tol,rho=rho,verb=verb)
end


function adaptquadinv(g::T,a::Number,b::Number; atol=0.0,rtol=0.0,maxevals=1e7,rho=1.0,verb=0) where T<:Function
"""
    I, E, segs, numevals = adaptquadinv(g,a::Real,b::Real;...
                                      atol=0.0,rtol=0.0,maxevals=1e7,rho=1.0,
    verb=0)

    1D adaptive pole-subtracting Gauss-Kronrod quadrature of 1/g, where `g`
    is a given smooth function, over (a,b).
    'rho' sets the max Bernstein ellipse parameter for dealing with poles.
    `atol` has precendence over 'rtol' in setting target accuracy.
    Based on miniguadgk.
    Scalar function g, for now.
"""
    if atol==0.0          # simpler logic than QuadGK. atol has precedence
        if rtol>0.0
            @assert rtol >= 1e-16
        else
            rtol = 1e-6   # default
        end
    end        
    r = gkrule()       # make a default panel rule
    n = length(r.gw)   # num embedded Gauss nodes, overall "order" n
    V = r.x.^(0:2n)'   # Vandermonde
    fac = lu(V)        # factor it only once
    numevals = 2n+1
    mid, sca = (b+a)/2, (b-a)/2
    gvals = Vector{ComplexF64}(undef,2n+1)
    gvals = map(x -> g(mid + sca*x), r.x)  # (see miniquadgk for reason)
    ginvals = 1.0./gvals                   # also allocs
    segs = applypolesub!(gvals,ginvals,a,b,r,rho=rho,verb=verb,fac=fac)    # kick off adapt via mother seg
    #println(segs)
    I, E = segs.I, segs.E          # keep global estimates which get updated
    segs = [segs]                  # heap needs to be Vector
    while E>atol && E>rtol*abs(I) && numevals<maxevals
        s = heappop!(segs, Reverse)            # get worst seg
        (verb>0) && @printf "adaptquadinv tot E=%.3g, splitting (%g,%g) of meth=%d:\n" E s.a s.b s.meth
        split = (s.b+s.a)/2
        mid, sca = (split+s.a)/2, (split-s.a)/2
        gvals = map(x -> g(mid + sca*x), r.x)
        ginvals = 1.0./gvals
        s1 = applypolesub!(gvals,ginvals, s.a,split, r,rho=rho,fac=fac,verb=verb)
        #println(s1)
        mid, sca = (s.b+split)/2, (s.b-split)/2
        gvals = map(x -> g(mid + sca*x), r.x)
        ginvals = 1.0./gvals
        s2 = applypolesub!(gvals,ginvals, split,s.b, r,rho=rho,verb=verb,fac=fac)
        #println(s2)
        numevals += 2*(2n+1)
        I += -s.I + s1.I + s2.I    # update global integral and err
        E += -s.E + s1.E + s2.E
        heappush!(segs, s1, Reverse)
        heappush!(segs, s2, Reverse)
    end
    return I, E, segs, numevals
end

function applypolesub!(gvals::AbstractArray, ginvals::AbstractArray, a::Number,
                      b::Number, r::gkrule; rho=1.0, verb=0, fac=nothing)
# pole-correcting version of applygkrule. Changes the input ginvals array.
# no g eval; just pass in all vals and inverse vals (thinking to matrix case)
# Barnett 6/30/23
    @assert length(gvals)==length(ginvals)
    s = applygkrule(ginvals,a,b,r)   # create Segment w/ plain GK ans for (a,b)
    # now work in local coords wrt std seg [-1,1]...
    # get roots, g'(roots)  ...n seems good max # roots to pole-sub @ 2n+1 pts
    zr, dgdt = find_near_roots(gvals,r.x,rho=rho,fac=fac)
    (verb>0) && @printf "\tapplypole sub (%g,%g):\t%d roots\n" a b length(zr)
    if length(zr)==0 || length(zr)>length(r.gw)
        return s        # either nothing to do, or don't pole-sub too much!
    end
    Ipoles = zero(I)
    #println("before corr, ginvals = ",ginvals)
    for (i,z) in enumerate(zr)     # loop over roots of g, change user's ginvals
        Res_ginv = 1.0/dgdt[i]     # residue of 1/g
        (verb>0) && println("\t   pole #",i," z=",z) #,",  res =",Res_ginv)
        #println("pole vals = ",Res_ginv./(r.x .- z))
        @. ginvals -= Res_ginv/(r.x - z)    # subtract each pole off 1/g vals
        Ipoles += Res_ginv * log((1.0-z)/(-1.0-z))  # add analytic pole integral
        #println("ginvals = ",ginvals)
        #println("Ipoles = ",Ipoles)
    end
    sp = applygkrule(ginvals,a,b,r)   # GK on corr 1/g vals, another Segment ***
    (verb>0) && println("\t", sp.E, "(meth=2) <? ", s.E, "(plain GK)")
    if sp.E > s.E
        return s                # error not improved, revert to plain GK
    else
        sca = (b-a)/2
        # sp.meth = 2           # pole sub worked, record it... no, immutable!
        # sp.I += sca*Ipoles    # add (scaled) effect of poles
        # return sp             # that idea failed :(
        return Segment(a,b, sp.I+sca*Ipoles, sp.E, 2)  # immutable => new seg :(
    end
end
