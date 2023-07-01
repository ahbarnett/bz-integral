function find_near_roots(vals::Vector, nodes::Vector; rho=1.0, fac=nothing)
    """
    roots, derivs = find_near_roots(vals, nodes; rho=1.0, fac=nothing)

    Returns complex-valued roots of unique polynomial approximant
    g(z) matching the vector of `vals` at the vector `nodes`.
    The nodes are assumed to be well-chosen for interpolation on [-1,1].
    'roots' are returned in order of increasing (Bernstein) distance from
    the interval [-1,1]. Also returns 'derivs', the values of g' at each
    root.

    `rho > 0.0` sets the Bernstein ellipse parameter within which to keep
    roots. Recall that the ellipse for the standard segment `[-1,1]` has
    semiaxes `cosh(rho)` and `sinh(rho)`.

    To do: compare Boyd version using Cheby points (needs twice the degree)

    Alex Barnett 6/29/23.
    """
    n = length(nodes)
    if isnothing(fac)
        V = nodes.^(0:n-1)'  # Vandermonde
        c = V \ vals         # solve monomial coeffs (4us for 15 nodes)
    else
        c = fac \ vals       # solve via passed-in LU factorization of V (1.5us)
    end
    roots = PolynomialRoots.roots(c)       # find all roots (10us)
    #roots = PolynomialRoots.roots5(c[1:6])   # find roots only degree-5 (4us)
    # solve roots = (t+1/t)/2 to get t (Joukowsky map) values
    # (3us for middle bit...)
    t = roots .+ sqrt.(roots.^2 .+ 1)
    rhos = abs.(log.(abs.(t)))        # Bernstein param for each root
    nkeep = sum(rhos .< rho)          # then keep t with e^-rho < t < e^rho
    inds = sortperm(rhos)[1:nkeep]    # indices to keep
    roots = roots[inds]
    derivs = zero(roots)              # initialize deriv vals
    # following 4us per root
    for (i,r) in enumerate(roots)
        derc = c[2:end] .* (1:n-1)    # coeffs of deriv of poly
        derivs[i] = horner(r,derc...) # eval at root (** fix speed n-template?)
    end
    return roots, derivs
end

# Poly eval (from SGJ's 2019 JuliaCon talk)
# see 27 mins into: https://www.youtube.com/watch?v=mSgXWpvQEHE
horner(x::Number)=zero(x)
horner(x::Number,p1::Number)=p1
horner(x::Number,p1::Number,p...)=muladd(x,horner(x,p...),p1)  # labeled splat p
# handle coeff vector rather than list of args, by splat...  slows it down :(
horner(x::Number,c::Vector)=horner(x,c...)
# handle array arg x...
horner(x::AbstractArray,args...) = map(y -> horner(y,args...), x)
# Timing expts...
#a = rand(32); x = rand()
#julia> @btime horner($x,$a);
#  17.944 μs (654 allocations: 13.97 KiB)      <- splatting each time = bad
#julia> @btime horner($x,$(a...));
#  2.083 ns (0 allocations: 0 bytes)
