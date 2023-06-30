module Int1DBZ
"""
Int1DBZ: 1D Brillouin zone integration, comparing a variety of methods.

A H Barnett, June 2023. Based off simpler subset of Cont1DBZ, QuadGK
"""

using OffsetArrays
using Printf

using LoopVectorization    # for @avx in evaluators
export evalh, evalh_ref, fourier_kernel
include("evaluators.jl")

using QuadGK         # SGJ
export realadap, realadap_lxvm, realmyadap
include("integrators.jl")

using DataStructures
import Base.Order.Reverse
using Gnuplot
export Segment, gkrule, applygkrule, applygkrule!, miniquadgk, plot
include("miniquadgk.jl")

using PolynomialRoots     # low-order faster roots
using LinearAlgebra
export find_near_roots, horner
include("complex.jl")

end # module Int1DBZ
