function find_near_roots(vals::Vector, nodes::Vector; rho=1.0)
    """
    roots, derivs = find_near_roots(vals, nodes; rho=1.0)

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
    V = nodes.^(0:n-1)'   # Vandermonde
    c = V\vals            # solve monomial coeffs  *** to do precomp factor V
    roots = PolynomialRoots.roots(c)       # find all roots
    # solve roots = (t+1/t)/2 to get t (Joukowsky map) values
    t = roots .+ sqrt.(roots.^2 .+ 1)
    rhos = abs.(log.(abs.(t)))        # Bernstein param for each root
    nkeep = sum(rhos .< rho)          # then keep e^-rho < t < e^rho
    inds = sortperm(rhos)[1:nkeep]       # indices to keep
    roots = roots[inds]
    derivs = zero(roots)
    for (i,r) in enumerate(roots)
        derivs[i] = 0.0   # *** TO DO
    end
    return roots, derivs
end

#hornerpoly
#hornerpolyd

