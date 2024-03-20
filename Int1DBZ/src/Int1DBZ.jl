module Int1DBZ
"""
Int1DBZ: 1D Brillouin zone integration, comparing a variety of methods.

A H Barnett. started June 2023, based off simpler subset of Cont1DBZ, QuadGK
"""

# stuff all included code lumps need
using OffsetArrays
using Printf

# now group I/O by code lumps (must be in correct order)...

using LoopVectorization    # for @avx in evaluators
export evalh, evalh_ref, fourier_kernel
include("evaluators.jl")

using DataStructures
import Base.Order.Reverse
using Gnuplot
using CairoMakie
Gnuplot.options.gpviewer=true    # for vscode; see https://discourse.julialang.org/t/gnuplot-from-vscode-no-plot/65458/4
export Segment, gkrule, applygkrule, applygkrule!, miniquadgk, plotsegs!, showsegs!
include("miniquadgk.jl")

using PolynomialRoots     # low-order faster roots
using AMRVW               # reliable roots in O(N^2)
using NonlinearEigenproblems   # n>1 matrix case, NEP and PEP solvers
using LinearAlgebra
export find_near_roots, BZ_denominator_roots, roots_companion, few_poly_roots
include("complex.jl")
export qpade_sqrtsings     # middle integral (2D case, moving beyond 1D BZ)
export genchebquad
include("genchebquad.jl")

using QuadGK         # SGJ Pkg
export realadap, realadap_lxvm, realmyadap, applypolesub!, adaptquadinv
export realquadinv
include("integrators.jl")

# other utils
using Colors
include("z2color.jl")
export z2color
include("ellipk.jl")
export ellipkAGM

end # module Int1DBZ
