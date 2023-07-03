module Int1DBZ
"""
Int1DBZ: 1D Brillouin zone integration, comparing a variety of methods.

A H Barnett, June 2023. Based off simpler subset of Cont1DBZ, QuadGK
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
export Segment, gkrule, applygkrule, applygkrule!, miniquadgk, plot
include("miniquadgk.jl")

using PolynomialRoots     # low-order faster roots
using AMRVW               # reliable roots in O(N^2)
using LinearAlgebra
export find_near_roots, horner, shifted_fourier_series_roots
include("complex.jl")

using QuadGK         # SGJ Pkg
export realadap, realadap_lxvm, realmyadap, applypolesub!, adaptquadinv
export realquadinv
include("integrators.jl")

end # module Int1DBZ
