"""
YMaze is a software package to load recorded behaviour of fruit flies in a Y-Maze with location-dependent shocks and visualize it.
The data is recorded by Riddha Manna and Ana Marija Jaksic https://www.epfl.ch/labs/jaksic-lab.

"""
module YMaze

using Random, CairoMakie, CSV, RollingFunctions, DataFrames, CategoricalArrays, Statistics
using DocStringExtensions

include("maze.jl")
include("encoders.jl")
include("plotting.jl")
include("io.jl")

end # module
