### bare bones scalar Gauss-Kronrod quadrature
### so don't have to constantly hack or grok QuadGK. Mostly from QuadGK

# integration segment (a,b), estimated integral I, and estimated error E,
# and a record of method used.
# Segment is templated by 3 types, that of (a,b), that of I, and that of E,
# which it seems to read from the types of the constructor (a,b,I,E):
struct Segment{TX,TI,TE}
    a::TX
    b::TX
    I::TI
    E::TE
    meth::Int64
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


struct gkrule
    """
    gkrule: a Gauss-Kronrod rule for (-1,1).
    `x` is all 2n+1 nodes, `w` is all 2n+1 weights for Kronrod rule,
    'gw' is n weights for Gauss rule.
    Since n small, not worth desymmetrizing.
    """
    x::Vector{Float64}
    w::Vector{Float64}
    gw::Vector{Float64}
end
# default: get all nodes from SGJ's desymmetrized nodes, just n=7 odd case
gkrule() = gkrule([xd7;-xd7[end-1:-1:1]],[wd7;wd7[end-1:-1:1]],[gwd7;gwd7[end-1:-1:1]])

#function quadinvanal(fvals::Vector{Float64}, rho, ...)
#  return I,E
#end

function applyrule(f,a::Float64,b::Float64,r::gkrule;rho=0.0)
    # Eval func f and GK-rule to estimate integral on (a,b) and its error.
    # returns a Segment (as with SGJ's evalrule) containing:
    #  I estimated integral, and E estimated error.
    # Include various experimental methods here since that's where fvals
    # avail. Choose the min error between methods.
    mid, sca = (b+a)/2, (b-a)/2
    fvals = f.(mid .+ sca*r.x)
    Ig = sca * sum(fvals[2:2:end] .* r.gw)
    Ik = sca * sum(fvals .* r.w)
    E = abs(Ig-Ik)
    meth = 1
    # *** if rho>0, replace E with quadinvanal...    using fvals
    return Segment(a,b,Ik,E,meth)
end

function miniquadgk(f,a,b;atol=0.0,rtol=0.0,maxevals=1e7)
    # simple implement based on QuadGK, but easy to understand/modify.
    # specific to Float64 scalar function f.
    if atol==0.0          # simpler logic than QuadGK! atol has precedence
        if rtol>0.0
            @assert rtol >= 1e-16
        else
            rtol = 1e-6   # default
        end
    end        
    r = gkrule()   # default
    n = length(r.gw)   # num embedded Gauss nodes, overall "order" n
    numevals = 2n+1
    segs = applyrule(f,a,b,r)      # kick off adapt via eval mother seg
    I, E = segs.I, segs.E          # global estimates which get updated
    segs = [segs]                  # heap needs to be Vector
    while E>atol && E>rtol*abs(I) && numevals<maxevals
        s = heappop!(segs, Reverse)            # get worst seg
        mid, sca = (s.b+s.a)/2, (s.b-s.a)/2    # split it...
        s1 = applyrule(f,s.a,mid,r)
        s2 = applyrule(f,mid,s.b,r)
        numevals += 2*(2n+1)
        I += -s.I + s1.I + s2.I    # update global integral and err
        E += -s.E + s1.E + s2.E
        heappush!(segs, s1, Reverse)
        heappush!(segs, s2, Reverse)
    end
    # [resum as SGJ?]
    return I, E, segs, numevals
end
