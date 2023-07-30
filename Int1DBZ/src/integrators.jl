####### Conventional real-axis quadrature methods using QuadGK directly...

"""
    A = realadap(hm,ω,η;tol,verb)

    use QuadGK on Re axis to integrate 1/(ω - h(x) + iη), where h(x)
    is 2π-periodic with Fourier series coefficients `hm`, which are
    either scalar- or matrix-valued, and in a OffsetVector with indices
    -M:M. `tol` controls `rtol`.
"""
function realadap(hm,ω,η; tol=1e-8, verb=0, kernel=evalh_ref)
    # integrand (quadgk gives x a number; note I is Id if StaticArray matrix)
    f(x::Number) = tr(inv(complex(ω,η)*I - kernel(hm,x)))
    # note how `kernel` lets the evaluator be general
    if verb>0
        A,err,fevals = quadgk_count(f,0,2π,rtol=tol)
        @printf "\trealadap:\tfevals=%d,  claimed err=%.3g\n" fevals err
    else
        A,err = quadgk(f,0,2π,rtol=tol)
    end
    A
end

"""
    A = realadap_lxvm(hm,ω,η;tol,verb)

    Version of `realadap` using faster non-allocating Fourier series
    evaluator. By LXVM.
"""
realadap_lxvm(hm, ω, η; tol=1e-8, verb=0) =
    realadap(hm, ω, η; tol=tol, verb=verb, kernel=fourier_kernel)


"""
    A = realmyadap(hm,ω,η; tol=1e-8, ab=[0.0,2π], verb=0)

    use adaptive quad on Re axis to integrate 1/(ω - h(x) + iη), where h(x)
    is 2π-periodic with Fourier series coefficients `hm`, which are
    either scalar- or matrix-valued, and in a OffsetVector with indices
    -M:M. `tol` controls `rtol`.
    `ab` a 2-element real vector sets custom integration interval.
    `verb` greater than 0 gives debug info.

    Note: uses AHB's miniquadgk, as hinted by `my` in function name.
"""
function realmyadap(hm,ω,η; tol=1e-8, ab=[0.0,2π], verb=0)
    # as in realadap, tr is Trace. For scalar: tr(inv(..)) is reciprocal
    f(x::Number) = tr(inv(complex(ω,η)*I - fourier_kernel(hm,x)))
    A, E, segs, fevals = miniquadgk(f,ab[1],ab[2],rtol=tol)
    verb>0 && @printf "\trealmyadap:\tfevals=%d, nsegs=%d, claimed err=%.3g\n" fevals length(segs) E
    return A, E, segs, fevals
end


########### New methods more specific to reciprocal of analytic function...

"""
    A = realquadinv(hm,ω,η; tol=1e-8, ab=[0.0,2π], verb=0, rho=1.0)

    use adaptive pole-subtracting method (adaptquadinv) on Re axis to
    integrate 1/(ω - h(x) + iη), where h(x)
    is 2π-periodic with Fourier series coefficients `hm`, which are
    either scalar- or matrix-valued, and in a OffsetVector with indices
    -M:M.
    `tol` controls `rtol`.
    `ab` a 2-element real vector sets custom integration interval.
    `verb` greater than 0 gives debug info.
    `rho` is passed to adaptquadinv.
"""
function realquadinv(hm,ω,η; tol=1e-8, rho=1.0, verb=0, ab=[0.0,2π], rootmeth="PR")
    # scalar reciprocal of trace of inverse is analytic
    g(x::Number) = 1.0 / tr(inv(complex(ω,η)*I - fourier_kernel(hm,x)))
    return adaptquadinv(g,ab[1],ab[2], rtol=tol, rho=rho, verb=verb,rootmeth=rootmeth)
end


function adaptquadinv(g::T,a::Number,b::Number; atol=0.0,rtol=0.0,maxevals=1e7,rho=1.0,verb=0,rootmeth="PR") where T<:Function
"""
    I, E, segs, numevals = adaptquadinv(g, a::Real, b::Real; ...
                             atol=0.0,rtol=0.0,maxevals=1e7,rho=1.0,verb=0,
                             rootmeth="PR")

    1D adaptive pole-subtracting Gauss-Kronrod quadrature of 1/g,
    where `g` is a given locally analytic scalar function, over interval
    (a,b). It also handles `g` merely meromorphic.

    'rho' sets the max Bernstein ellipse parameter for dealing with poles.
    As in QuadGK, `atol` has precendence over 'rtol' in setting target
    accuracy, and `maxevals` limits the number of `g` evals.
    `verb`=1 gives debug text output, 2 gives segment-level, 3 pole-sub-level
    `rootmeth` controls poly root-finding method (see `find_near_roots()`).

    Notes: Based on miniguadgk.
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
    mid0, sca0 = (b+a)/2, (b-a)/2
    gvals = map(x -> g(mid0 + sca0*x), r.x)   # allocs, once
    ginvals = 1.0./gvals
    # kick off adapt via mother seg...
    seg = applypolesub!(gvals,ginvals,a,b,r,rho=rho,verb=verb-2,fac=fac,rootmeth=rootmeth)
    I, E = seg.I, seg.E          # keep global estimates which get updated
    segs = [seg]                  # heap needs to be Vector
    while E>atol && E>rtol*abs(I) && numevals<maxevals
        s = heappop!(segs, Reverse)            # get worst seg
        verb>1 && @printf "adaptquadinv tot E=%.3g, splitting (%g,%g) of npoles=%d:\n" E s.a s.b s.npoles
        split = (s.b+s.a)/2
        mid, sca = (split+s.a)/2, (split-s.a)/2
        map!(x -> g(mid + sca*x), gvals, r.x)
        # for (i,x) in enumerate(r.x); gvals[i] .= g(mid + sca*r.x[i]); end  # loop so no alloc? No, was just the same... due to g() itself??
        ginvals .= 1.0./gvals                    # math cheap; in-place
        s1 = applypolesub!(gvals,ginvals, s.a,split, r,rho=rho,fac=fac,verb=verb-2,rootmeth=rootmeth)
        mid, sca = (s.b+split)/2, (s.b-split)/2
        map!(x -> g(mid + sca*x), gvals, r.x)
        ginvals .= 1.0./gvals
        s2 = applypolesub!(gvals,ginvals, split,s.b, r,rho=rho,verb=verb-2,fac=fac,rootmeth=rootmeth)
        numevals += 2*(2n+1)
        I += -s.I + s1.I + s2.I    # update global integral and err
        E += -s.E + s1.E + s2.E
        heappush!(segs, s1, Reverse)
        heappush!(segs, s2, Reverse)
    end
    if verb>0
        @printf "\tadaptquadinv:\tfevals=%d, nsegs=%d, claimed err=%.3g\n" numevals length(segs) E
        npolesegs = zeros(Int,6)   # count segs of each type (plain, 1-pole...)
        for m=1:5, npolesegs[m] = sum([s.npoles==m-1 for s in segs]); end
        npolesegs[6] = sum([s.npoles>4 for s in segs])
        @printf "\t\t\t%d plain GK segs; 1,2,..-pole segs [%d %d %d %d]; >4-pole %d\n" npolesegs[1] npolesegs[2] npolesegs[3] npolesegs[4] npolesegs[5] npolesegs[6]
    end
    return I, E, segs, numevals
end

function applypolesub!(gvals::AbstractArray, ginvals::AbstractArray, a::Number,
                      b::Number, r::gkrule; rho=1.0, verb=0, fac=nothing, rootmeth="PR", maxpolesubint=Inf)
# pole-correcting version of applygkrule. Changes the input ginvals array.
# no local g evals; just pass in all vals and reciprocal vals.
# Barnett 6/30/23
    @assert length(gvals)==length(ginvals)
    s = applygkrule(ginvals,a,b,r)   # create Segment w/ plain GK ans for (a,b)
    if b-a > maxpolesubint           # save a few pole-sub considerations? no
        return s
    end
    # now work in local coords wrt std seg [-1,1]...
    # get roots, g'(roots)  ...n seems good max # roots to pole-sub @ 2n+1 pts
    zr, dgdt = find_near_roots(gvals,r.x,rho=rho,fac=fac,meth=rootmeth)
    verb>0 && @printf "\tapplypole sub (%g,%g):\t%d roots\n" a b length(zr)
    if length(zr)==0 || length(zr)>4  #length(r.gw)  # or 3, captures most
        return s        # either nothing to do, or don't pole-sub too much!
    end
    Ipoles = zero(I)
    #println("before corr, ginvals = ",ginvals)
    for (i,z) in enumerate(zr)     # loop over roots of g, change user's ginvals
        Res_ginv = 1.0/dgdt[i]     # residue of 1/g
        verb>0 && println("\t   pole #",i," z=",z) #,",  res =",Res_ginv)
        #println("pole vals = ",Res_ginv./(r.x .- z))
        @. ginvals -= Res_ginv/(r.x - z)    # subtract each pole off 1/g vals
        Ipoles += Res_ginv * log((1.0-z)/(-1.0-z))  # add analytic pole integral
        #println("ginvals = ",ginvals)
        #println("Ipoles = ",Ipoles)
    end
    sp = applygkrule(ginvals,a,b,r)   # GK on corr 1/g vals, another Segment ***
    verb>0 && println("\t", sp.E, "(pole-sub) <? ", s.E, "(plain GK)")
    if sp.E > s.E
        return s                # error not improved, revert to plain GK
    else
        sca = (b-a)/2
        # sp.meth = length(zr)  # pole sub worked, record it... no, immutable!
        # sp.I += sca*Ipoles    # add (scaled) effect of poles
        # return sp             # that idea failed :(
        return Segment(a,b, sp.I+sca*Ipoles, sp.E, length(zr))  # immutable => new seg :(
    end
end
