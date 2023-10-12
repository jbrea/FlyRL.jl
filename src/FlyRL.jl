"""
FlyRL is a software package to fit, simulate and analyze reinforcement learning methods to
the navigation behaviour of fruit flies in a Y-Maze with location-dependent shocks.
The data is recorded by Riddha Manna and Ana Marija Jaksic https://www.epfl.ch/labs/jaksic-lab.
"""
module FlyRL

using StatsBase, CategoricalArrays, DataFrames, Printf, LinearAlgebra
using CSV, RollingFunctions, Serialization
using StaticArrays, ComponentArrays
using Enzyme, Optim, NLopt
using Random
using DocStringExtensions
using YMaze
import YMaze: encode, encode!, AbstractEncoder, levels, labels,
              ArmEncoder, ShockArmEncoder, ColorEncoder, plot_track,
              plot_maze, read, read_directory, DEFAULT_Δt, with_outliers,
              in_maze, in_left, in_right, in_middle, in_turn,
              random_maze_position, hascols, getcols, SQRT3

include("plotting.jl")
include("io.jl")

include("encoders.jl")
include("summary_stats.jl")
include("preprocessor.jl")

include("mlp.jl")
include("models.jl")
include("simulators.jl")

include("fitting.jl")

end # module
