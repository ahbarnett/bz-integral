function applyrule!(fwrk,f,a::Float64,b::Float64,r::gkrule;rho=0.0)
    # Eval func f and GK-rule to estimate integral on (a,b) and its error.
    # fwrk is preallocated workspace (must be size>=2n+1, n=length(r.gw)).
    #  ...That is an expt in C-style static workspace, not very happy with.
    #  (why can't have local one-off static allocation of 32 CF64's?)
    #
    # Returns a Segment (as with SGJ's evalrule) containing:
    #  I estimated integral, and E estimated error.
    # Include various experimental methods here since that's where fvals
    # avail. Choose the min error between methods. Barnett 6/28/23
    mid, sca = (b+a)/2, (b-a)/2
    n = length(r.gw)
    
    #= version using the prealloc fwrk... still have 100 bytes/feval alloc ???
    for j=1:2n+1     # no alloc
        fwrk[j] = f(mid + sca*r.x[j])        
    end
    #fwrk[1:2n+1] = f.(mid .+ sca*r.x)       # fill the prealloc vector ... doesn't do it, allocs 96 bytes (6 CF64's)
    @views Ig = sca * sum(fwrk[2:2:2n+1] .* r.gw)   # allocs 11 CF64s
    @views Ik = sca * sum(fwrk[1:2n+1] .* r.w)      # allocs 19 CF64s
    =#

    # the clean code I'd like to write... does 4x the alloc of above :(
    #fwrk[1:2n+1] = f.(mid .+ sca*r.x)       # fill the prealloc vector ... doesn't do it
    #Ig = sca * sum(fwrk[2:2:end] .* r.gw)
    #Ik = sca * sum(fwrk .* r.w)

    # less dicking around but still alloc-free hand loop, using fixed fwrk array
    Ik = Ig = zero(fwrk[1])   # 0 of type of els of fwrk
    for j=1:2n+1
        fwrk[j] = f(mid + sca*r.x[j])        # save the evals for Gauss below
        Ik += fwrk[j]*r.w[j]
    end
    for i=1:n            # do Gauss quadr
        Ig += fwrk[2i]*r.gw[i]
    end
    Ik *= sca
    Ig *= sca
    
    #= # OR dicking around to avoid a simple static array allocation for fvals!
    #Ig = Ik = zero(F)  failed   (using f::F  .... where F)
    j=1
    Ik = r.w[j] * f(mid + sca*r.x[j])     # j=1 hack to get type of Ik,Ig right
    Ig = 0.0*Ik                            # otherwise how know f's range type?
    for halfj=1:n    # loop so no storage of func values ...
        j = 2halfj   # ... this cannot be the future of scientific computing :(
        fj = f(mid + sca*r.x[j])
        Ik += r.w[j] * fj
        Ig += r.gw[halfj] * fj     
        j = 2halfj+1
        Ik += r.w[j] * f(mid + sca*r.x[j])
    end              # still allocates about 100 bytes per eval ... why?
    Ik *= sca
    Ig *= sca
    =#

    E = abs(Ig-Ik)
    meth = 1

    if (rho>0.0)
        # replace Ik,E with quadinvanal...    using fvals
        # *** TO DO
    end
    return Segment(a,b,Ik,E,meth)
end

#=     set up easy-to use version doing own alloc ?
function applyrule(f,a,b,r,...)
    wrk = 
    applyrule!(wrk, f,a,b,r...) =    *** TO DO
=#
