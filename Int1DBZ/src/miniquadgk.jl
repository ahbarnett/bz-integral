# Bare bones scalar Gauss-Kronrod quadrature, with segment diagnosis,
# so don't have to constantly hack or grok QuadGK.

# integration segment (a,b), estimated integral I, and estimated error E.
# Segment is templated by 3 types, that of (a,b), that of I, and that of E,
# which it seems to read from the types of the constructor (a,b,I,E):
struct Segment{TX,TI,TE}
    # fields as in QuadGK...
    a::TX
    b::TX
    I::TI
    E::TE
    # fields to do with experimental methods...
    npoles::Int64       # number of poles subtracted from integrand
end
# make segments sort by error in a heap...
Base.isless(i::Segment, j::Segment) = isless(i.E, j.E)

# precomputed n=7 rule in double precision (computed in 100-bit arithmetic),
# since this is the common case.
# AHB: these are just non-positive nodes and corresp weights
# Gauss rule is order 2n-1=13, Kronrod is order 3n+1=22.
# K order?: 2*(2n+1)-1 -n, since 2n+1 nodes, but satisfy n extra constraints
const xd7 = [-9.9145537112081263920685469752598e-01,
             -9.4910791234275852452618968404809e-01,
             -8.6486442335976907278971278864098e-01,
             -7.415311855993944398638647732811e-01,
             -5.8608723546769113029414483825842e-01,
             -4.0584515137739716690660641207707e-01,
             -2.0778495500789846760068940377309e-01,
             0.0]
const wd7 = [2.2935322010529224963732008059913e-02,
             6.3092092629978553290700663189093e-02,
             1.0479001032225018383987632254189e-01,
             1.4065325971552591874518959051021e-01,
             1.6900472663926790282658342659795e-01,
             1.9035057806478540991325640242055e-01,
             2.0443294007529889241416199923466e-01,
             2.0948214108472782801299917489173e-01]
const gwd7 = [1.2948496616886969327061143267787e-01,
              2.797053914892766679014677714229e-01,
              3.8183005050511894495036977548818e-01,
              4.1795918367346938775510204081658e-01]


"""
    gkrule: a Gauss-Kronrod rule for (-1,1).
    `x` is all 2n+1 nodes, `w` is all 2n+1 weights for Kronrod rule,
    'gw' is n weights for Gauss rule.
    Since n small, not worth desymmetrizing. Simplicity wins.
"""
struct gkrule
    x::Vector{Float64}
    w::Vector{Float64}
    gw::Vector{Float64}
end
# default: get all nodes from SGJ's desymmetrized nodes, just n=7 odd case
gkrule() = gkrule([xd7;-xd7[end-1:-1:1]],[wd7;wd7[end-1:-1:1]],[gwd7;gwd7[end-1:-1:1]])

function applygkrule(f::T,a::Float64,b::Float64,r::gkrule) where T<:Function
# convenience wrapper which allocates own fvals array (hence slow), then fills
    mid, sca = (b+a)/2, (b-a)/2
    fvals = f.(mid .+ sca*r.x)      # eval f, allocates
    return applygkrule(fvals,a,b,r)
end

function applygkrule!(fvals::AbstractArray,f::T,a::Float64,b::Float64,r::gkrule) where T<:Function
# as applygkrule(fvals..) but uses fvals as workspace and overwrites it
    mid, sca = (b+a)/2, (b-a)/2
    #fvals .= f.(mid .+ sca*r.x)      # note .= to prevent reallocation ?
    for j in eachindex(r.x)
        fvals[j] = f(mid + sca*r.x[j])
    end
    return applygkrule(fvals,a,b,r)
end

"""
    seg = applygkrule(fvals::AbstractArray,a::Float64,b::Float64,r::gkrule)
    
    Use `fvals` as function values at the nodes `r.x` to apply plain
    Gauss-Kronrod quadrature on interval (a,b). Returns `Segment` object
    containing `a` and `b` endpoints, `I` integral estimate, `E` error
    estimate, and `npoles` number of poles subtracted (0 for plain GK).
"""
function applygkrule(fvals::AbstractArray,a::Float64,b::Float64,r::gkrule)
    # Barnett 6/30/23 tidying up applyrule!
    n = length(r.gw)
    Ik = Ig = zero(fvals[1])   # 0 of type of els of fwrk
    for j=1:2n+1               # do Kronrod quadr
        Ik += fvals[j]*r.w[j]
    end
    for j=1:n                  # do Gauss quadr using even nodes
        Ig += fvals[2j]*r.gw[j]
    end
    sca = (b-a)/2
    Ik *= sca
    Ig *= sca
    E = abs(Ig-Ik)
    return Segment(a,b,Ik,E,0)   # 0 is npoles, means plain GK
end

"""
    I, E, segs, numevals = miniquadgk(f,a::Real,b::Real;...
                                      atol=0.0,rtol=1e-6,maxevals=1e7)

    Simple implementation of 1D adaptive Gauss-Kronrod quadrature of function
    f over (a,b). `atol` has precendence over 'rtol' in setting target accuracy.
    Based on QuadGK, using same segment heap, but easy to understand/modify.
    Specific to Float or Complex scalar function f, for now.
"""
function miniquadgk(f,a::Real,b::Real; atol=0.0,rtol=1e-6,maxevals=1e7)
    if atol==0.0          # simpler logic than QuadGK. atol has precedence
        @assert rtol >= 1e-16
    end        
    r = gkrule()       # make a default panel rule
    n = length(r.gw)   # num embedded Gauss nodes, overall "order" n
    numevals = 2n+1
    mid, sca = (b+a)/2, (b-a)/2
    # fact that next two lines needed to match quadgk, not plain f.(..), lame:
    fvals = Vector{ComplexF64}(undef,2n+1) # plain fvals=f.(..) worse than this
    # is fvals alloc needed?
    fvals = map(x -> f(mid + sca*x), r.x)  # as good as explicit loop, no alloc
    # fvals .= f.(mid .+ sca*r.x)   # fill prealloc via .= but cause alloc slow!
    segs = applygkrule(fvals,a,b,r)      # kick off adapt via eval mother seg
    I, E = segs.I, segs.E          # keep global estimates which get updated
    segs = [segs]                  # heap needs to be Vector
    while E>atol && E>rtol*abs(I) && numevals<maxevals
        s = heappop!(segs, Reverse)            # get worst seg
        split = (s.b+s.a)/2
        s1 = applygkrule!(fvals,f,s.a,split,r)   # fvals is workspace (no alloc)
        s2 = applygkrule!(fvals,f,split,s.b,r)
        numevals += 2*(2n+1)
        I += -s.I + s1.I + s2.I    # update global integral and err
        E += -s.E + s1.E + s2.E
        heappush!(segs, s1, Reverse)
        heappush!(segs, s2, Reverse)
    end
    # to do *** resum as SGJ?
    return I, E, segs, numevals
end

"""
    plotsegs!(segs, session=:default) uses Gnuplot.jl to add a Segment or
    vector of such to the given gnuplot session (or start session if did not
    exist).
    Segment is a type from miniquadgk.
    Color-coding via npoles is used (a field only used outside miniquadgk).
"""
function plotsegs!(segs::Vector{Segment{TX,TI,TE}}, session=:default) where {TX,TI,TE}
    a = [s.a for s in segs]
    b = [s.b for s in segs]
    i = [s.npoles==0 for s in segs]     # inds of std GK segs
    ab = [a[i] b[i]]
    gpsesh = Gnuplot.options.default
    Gnuplot.options.default=session  # see https://github.com/gcalderone/Gnuplot.jl/issues/63
    @gp :- real(ab) imag(ab) "w lp pt 1 lc rgb '#000000' tit 'GK segs'"
    i = [s.npoles>0 for s in segs]     # pole-sub segs
    if sum(i)>0
        ab = [a[i] b[i]]
        @gp :- real(ab) imag(ab) "w lp pt 1 lc rgb '#00ff00' tit 'pole-sub'"
    end
    xmax = maximum([a;b])
    xmin = minimum([a;b])
    y0 = 0.3*(xmax-xmin)     # y range to view
    @gp :- "set size ratio -1" xrange=[xmin,xmax] yrange=[-y0,y0]
    Gnuplot.options.default=gpsesh        # restore prev session
end
plotsegs!(seg::Segment,args...) = plotsegs!([seg],args...)  # handle single segment

