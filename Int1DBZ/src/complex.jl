function find_near_roots(vals::Vector, nodes::Vector; rho=1.0, fac=nothing)
    """
    roots, derivs = find_near_roots(vals, nodes; rho=1.0, fac=nothing)

    Returns complex-valued roots of unique polynomial approximant g(z)
    matching the vector of `vals` at the vector `nodes`.  The nodes
    are assumed to be well-chosen for interpolation on [-1,1].
    'roots' are returned in order of increasing (Bernstein) distance
    from the interval [-1,1]. It also computes 'derivs', the
    corresponding values of g' at each kept root.

    `rho > 0.0` sets the Bernstein ellipse parameter within which to keep
    roots. Recall that the ellipse for the standard segment `[-1,1]` has
    semiaxes `cosh(rho)` horizontally and `sinh(rho)` vertically.

    `fac` allows user to pass in a pre-factorized (eg LU) object for
    the Vandermonde matrix. This accelerates things by 3us for 15 nodes.

    To do:
    1) template so compiles for known n (speed up roots? poly eval?)
    2) compare Boyd version using Cheby points (needs twice the degree)

    Alex Barnett 6/29/23 - 7/4/23
    """
    n = length(nodes)
    if isnothing(fac)
        V = nodes.^(0:n-1)'  # Vandermonde
        c = V \ vals         # solve monomial coeffs (4us for 15 nodes)
    else
        c = fac \ vals       # solve via passed-in LU factorization of V (1.5us)
    end
    roots = PolynomialRoots.roots(c)       # find all roots (typ 7-10us)
    #roots = PolynomialRoots.roots5(c[1:6])   # find roots only degree-5 (4us)
    #roots = AMRVW.roots(c)                # 10x slower (~100us)
    #roots = roots_companion(reverse(c))   # also 10x slower (~100us)
    #return roots, empty(roots)   # exit for speed test of c-solve + roots only
    # Solve roots = (t+1/t)/2 to get t (Joukowsky map) values (1 us)
    t = @. roots + sqrt(roots^2 - 1.0)
    rhos = abs.(log.(abs.(t)))        # Bernstein param for each root
    nkeep = sum(rhos .< rho)          # then keep t with e^-rho < t < e^rho
    inds = sortperm(rhos)[1:nkeep]    # indices to keep
    roots = roots[inds]
    derivs = zero(roots)              # initialize deriv vals
    derc = zero(c[1:end-1])           # alloc
    for (i,r) in enumerate(roots)     # (1us for 14 roots degree 14)
        derc .= c[2:end] .* (1:n-1)         # coeffs of deriv of poly, no alloc?
        derivs[i] = Base.evalpoly(r,derc)   # eval at root (14 ns)
    end
    return roots, derivs
end

function BZ_denominator_roots(hm::AbstractArray{T},ω,η) where T
    """
    x = BZ_denominator_roots(hm,ω,η)
    
    returns all (generally-complex) roots of scalar- or matrix-valued
    denominator function F(x) := (ω+iη)I-H(x), where H(x) is 2π-periodic with
    Fourier series coefficients an OffsetVector `hm` with indices
    -M:M.  Each coefficient is a scalar or StaticArray n*n matrix.
    There will be 2Mn roots returned. Their real parts will be in [0,2π).

    For n=1, Boyd's method is used: find roots of degree-2M polynomial in
    z=e^{ix}.
    For n>1, a polynomial EVP is solved for z values where F(z) singular.
    Reliability not speed is the goal here.
    """
    # Barnett 7/4/23
    hmplusc = -hm;                   # was copy(hm) else hmconst changes hm!
    hmplusc[0] += I*complex(ω,η)     # F series for denominator
    hmplusc_vec = hmplusc.parent     # shift powers by M: data vec inds 1:2M+1
    n = size(hm[0],1)                # dim (matrix size)
    if T<:Number
        z = AMRVW.roots(hmplusc_vec) # more reliable for M>30 than PRoots.roots
    else
        pep = PEP(Matrix.(hmplusc_vec))  # set up PEP; SMatrix -> plain Matrix
        z,V = polyeig(pep)               # slow? V is n*J stack of right evecs
    end
    x = @. log(z)/im                 # solve z = e^{ix} for roots x of denom
    x = @. mod(real(x),2π) + im*imag(x)  # fold Re to [0,2π), for humans
    return x
end

"""
    roots_companion(a)

    find all complex roots of polynomial a[1]*z^n + a[2]*z^(n-1) + ... + a[n+1]
    via companion matrix EVP in O(n^3) time. Similar to MATLAB roots.
    Note poly coeffs are in reverse order that in many Julia pkgs.
    If the entire C plane is a root, returns [complex(NaN)].

    Local reference implementation; superceded by other pkgs.
"""
function roots_companion(a::AbstractVector{<:Number})
    # does not allow dims>1 arrays
    a = complex(a)          # idempotent, unlike Complex{T} for T a type...
    T = eltype(a)
    while length(a)>1 && a[1]==0.0         # gobble up any zero leading coeffs
        a = a[2:end]
    end
    if isempty(a) || (a==[0.0])            # done, meaningless
        return [complex(NaN)]     # array, for type stability. signifies all C
    end
    deg = length(a)-1       # a is now length>1 with nonzero 1st entry
    if deg==0
        return T[]          # done: empty list of C-#s
    end
    a = reshape(a[deg+1:-1:2] ./ a[1],(deg,1))    # make monic, col and flip
    C = [ [zeros(T,1,deg-1); Matrix{T}(I,deg-1,deg-1)] -a ]   # stack companion mat
    # at this point we want case of real C to be possible, faster than complex
    complex(eigvals!(C))    # overwrite C, and we don't want the vectors
end
# Note re don't need evecs: see also LinearAlgebra.LAPACK.geev!
