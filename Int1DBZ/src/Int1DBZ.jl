module Int1DBZ
"""
Int1DBZ: 1D Brillouin zone integration, comparing a variety of methods.

A H Barnett, June 2023. Based off of simpler subset of Cont1DBZ.
"""

using OffsetArrays
using LinearAlgebra
using Printf
using PolynomialRoots     # low-order faster roots
using FourierSeriesEvaluators: fourier_contract     # LXVM

using LoopVectorization    # for @avx in evaluators
export evalh, evalh_ref, fourier_kernel
include("evaluators.jl")

using QuadGK         # SGJ
export realadap, realadap_lxvm
include("integrators.jl")

#include("miniquadgk.jl")


end # module Int1DBZ
