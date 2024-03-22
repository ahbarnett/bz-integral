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
    f(x::Number) = tr(inv(complex(ω,η)*LinearAlgebra.I - kernel(hm,x)))
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
    f(x::Number) = tr(inv(complex(ω,η)*LinearAlgebra.I - fourier_kernel(hm,x)))
    A, E, segs, fevals = miniquadgk(f,ab[1],ab[2],rtol=tol)
    verb>0 && @printf "\trealmyadap:\tfevals=%d, nsegs=%d, claimed err=%.3g\n" fevals length(segs) E
    return A, E, segs, fevals
end


########### New methods more specific to reciprocal of analytic function...

"""
    A = realquadinv(hm,ω,η; tol=1e-8, ab=[0.0,2π], verb=0, rho=exp(1))

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
function realquadinv(hm,ω,η; tol=1e-8, rho=exp(1), verb=0, ab=[0.0,2π], rootmeth="PR")
    # scalar reciprocal of trace of inverse is analytic
    g(x::Number) = 1.0 / tr(inv(complex(ω,η)*LinearAlgebra.I - fourier_kernel(hm,x)))
    return adaptquadinv(g,ab[1],ab[2], rtol=tol, rho=rho, verb=verb,rootmeth=rootmeth)
end


function adaptquadinv(g::T,a::Number,b::Number; atol=0.0,rtol=0.0,maxevals=1e7,rho=exp(1),verb=0,rootmeth="PR") where T<:Function
"""
    I, E, segs, numevals = adaptquadinv(g, a::Real, b::Real; ...
                             atol=0.0,rtol=0.0,maxevals=1e7,rho=exp(1),verb=0,
                             rootmeth="PR")

    1D adaptive pole-subtracting Gauss-Kronrod quadrature of 1/g,
    where `g` is a given locally analytic scalar function, over interval
    (a,b). It also handles `g` merely meromorphic.

    'rho>1' sets the max Bernstein ellipse parameter for dealing with poles.
    As in QuadGK, `atol` has precendence over 'rtol' in setting target
    accuracy, and `maxevals` limits the number of `g` evals.
    `verb`=1 gives debug text output, 2 gives segment-level, 3 pole-sub-level
    `rootmeth` controls poly root-finding method (see `find_near_roots()`).

    Notes: 1) Based on miniguadgk.
    2) Timings for filling gvals Vector twice in while-loop:
    gvals .= map(x -> g(mid+sca*x), r.x)    added 5us (AHB)
    map!(x -> g(mid+sca*x), gvals, r.x)     added 1us (LXVM)
    the explicit loop via enumerate is the baseline.
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
        mid, sca = (split+s.a)/2, (split-s.a)/2    # no reuse mid0 (type-instab)
        for (i,x) in enumerate(r.x); gvals[i] = g(mid + sca*r.x[i]); end   # faster than map! or map
        ginvals .= 1.0./gvals                    # math cheap; in-place
        s1 = applypolesub!(gvals,ginvals, s.a,split, r,rho=rho,fac=fac,verb=verb-2,rootmeth=rootmeth)
        mid = (s.b+split)/2
        for (i,x) in enumerate(r.x); gvals[i] = g(mid + sca*r.x[i]); end
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
                      b::Number, r::gkrule; rho=exp(1), verb=0, fac=nothing, rootmeth="PR", maxpolesubint=Inf)
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
    verb>0 && @printf "\tapplypolesub (%g,%g):\t%d roots\n" a b length(zr)
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


########### New methods specific to integrand analytic with 1/sqrt singularities...

function adaptquadsqrt(f::Function,a::Number,b::Number;
    atol=0.0,rtol=0.0,maxevals=1e7,rho=exp(1),verb=0)
    """
        I, E, segs, numevals = adaptquadsqrt(f::Function, a::Real, b::Real; ...
                                 atol=0.0,rtol=1e-6,maxevals=1e7,rho=exp(1),verb=0)
    
        1D adaptive Gauss-Kronrod quadrature on (a,b) of function `f`, handling 1/sqrt
        singularities near to a panel efficiently by a quadratic Pade
        plus generalized Chebyshev quadrature scheme. This is intended for the middle
        Brillouin zone integration in the iterated integration approach to
        Green's function or DOS calculations, where 1/sqrt is the generic singularity.
    
        'rho>1' sets the max Bernstein ellipse parameter for dealing with poles.
        As in QuadGK, `atol` has precendence over 'rtol' in setting target
        accuracy, and `maxevals` limits the number of function evaluations.
        `verb`=1 gives debug text output, 2 gives segment-level info, etc.
    
        Notes: 1) Based on miniguadgk (hence QuadGK) and adaptquadinv.
        2) no emphasis on timings yet, since `f` evals are assumed dominant.
        Barnett 3/22/24
    """
    if atol==0.0          # logic so if user sets atol>0, rtol left at 0
        if rtol>0.0
            @assert rtol >= 1e-16
        else
            rtol = 1e-6   # default
        end
    end
    r = gkrule()               # make a (for now default) panel rule
    qtol = max(atol,rtol)      # qpade tol, for each seg, hack for now
    # kick off adapt via mother seg...
    numevals = [0]             # counter (array since need to update in func)
    seg = applyqpade!(numevals,f,a,b,r,qtol, rho=rho,verb=verb-2)
    I, E = seg.I, seg.E           # keep global estimates which get updated
    segs = [seg]                  # heap needs to be Vector
    while E>atol && E>rtol*abs(I) && numevals[1]<maxevals    # atol takes precedence over rtol
        s = heappop!(segs, Reverse)            # get worst seg
        verb>1 && @printf "adaptquadsqrt tot E=%.3g, splitting (%g,%g) of nsqrt=%d:\n" E s.a s.b s.nsqrtsings
        split = (s.a+s.b)/2
        s1 = applyqpade!(numevals, f, s.a,split, r, qtol, rho=rho,verb=verb-2)
        s2 = applyqpade!(numevals, f, split,s.b, r, qtol, rho=rho,verb=verb-2)
        I += -s.I + s1.I + s2.I    # update global integral and err
        E += -s.E + s1.E + s2.E
        heappush!(segs, s1, Reverse)
        heappush!(segs, s2, Reverse)
    end
    if verb>0
        nsqrt = sum([s.nsqrtsings>0 for s in segs])
       @printf "\tadaptquadsqrt:\tfevals=%d, nsegs=%d (nsqrt=%d), claimed err=%.3g\n" numevals[1] length(segs) nsqrt E
    end
    return I, E, segs, numevals[1]
end
    
function applyqpade!(nev::Vector{Int}, f::Function, a::Number, b::Number, r::gkrule, tol::Real;
    rho=exp(1), verb=0, pGCQ=10)
    # Invsqrt-correcting version of applygkrule, for interval (a,b).
    # Evaluates function f at GK nodes in (a,b), uses QPade to locate any nearby
    # 1/sqrt singularities, and if one found, eval f again at GCQ nodes built
    # custom for its location (expensive). Updates (changes) nev[1], the counter
    # for f evaluations. pGCQ is max degree of func set for GCQ.
    # Barnett 3/22/24
    mid, sca = (b+a)/2, (b-a)/2
    fvals = @. f(mid + sca*r.x)    # needed shortly also for QPade
    s = applygkrule(fvals,a,b,r)   # create Segment w/ plain GK ans for (a,b)
    nev[1] += length(r.x)
    # now work in local coords wrt std seg [-1,1]. Find any sqrt-singularities...
    zs, dDdt = qpade_sqrtsings(fvals, r.x, rho=rho)
    verb>0 && println("\tapplyqpade (",a,",",b,"):\tzsings = ", mid .+ sca*zs)
    if length(zs)!=1
        verb>0 && @printf "\t\tdo not attempt sqrt-sing.\n"
        return s        # either nothing to do, or too many sings to handle!
    else
        z0 = zs[1]      # exactly 1 sqrt sing in Bernstein ellipse
        fs(x::Number) = reduce(vcat, x^k.*[1, reim(1/sqrt(x-z0))...] for k=0:pGCQ)
        xg, wg, _ = genchebquad(fs,-1.0,1.0,tol;verb=verb-1)       # build a GCQ
        ng = length(xg)
        verb>0 && @printf "\t\tGCQ built (z0=%.10g+%.10gi, tol=%g, %d nodes)\n" real(z0) imag(z0) tol ng
        fg = f.(mid .+ sca*xg)                 # do expensive f evals
        nev[1] += ng
        Is = sca*sum(fg .* wg)          # slow but who cares
        tol2 = 1e-2*tol            # slightly better acc to assess error
        xg2, wg2, _ = genchebquad(fs,-1.0,1.0,tol2;verb=verb-1)     # build a more acc GCQ
        ng2 = length(xg2)
        verb>0 && @printf "\t\tGCQ more acc built (tol2=%g, %d nodes)\n" tol2 ng2
        fg2 = f.(mid .+ sca*xg2)                 # do more expensive f evals
        nev[1] += ng2
        Is2 = sca*sum(fg2 .* wg2)
        E=abs(Is-Is2)
        verb>0 && println("\t\tcompare err estims: ", E, " (QPade+GCQ) <? ", s.E, " (plain GK)")
        if E > s.E
            verb>0 && @printf "\t\tno: fall back on plain GK...\n"
            return s                # error not improved, revert to plain GK
        else
            verb>0 && @printf "\t\tyes: accept QPade_GCQ...\n"
            return Segment(a,b,Is2,E, 0,length(zs))    # immutable => create new seg :(
        end
    end
end    

