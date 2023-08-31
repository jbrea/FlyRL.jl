"""
FlyRL is a software package to fit, simulate and analyze reinforcement learning methods to
the navigation behaviour of fruit flies in a Y-Maze with location-dependent shocks.
The data is recorded by Riddha Manna and Ana Marija Jaksic https://www.epfl.ch/labs/jaksic-lab.
"""
module FlyRL

using CairoMakie, StatsBase, CategoricalArrays, DataFrames, Printf, LinearAlgebra
using CSV, RollingFunctions, Serialization
using StaticArrays, ComponentArrays
using Enzyme, Optim, NLopt
using Random
using DocStringExtensions

include("plotting.jl")
include("io.jl")

include("maze.jl")
include("encoders.jl")
include("summary_stats.jl")
include("preprocessor.jl")

include("mlp.jl")
include("models.jl")
include("simulators.jl")

include("fitting.jl")

end # module
