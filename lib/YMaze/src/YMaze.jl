"""
YMaze is a software package to load recorded behaviour of fruit flies in a Y-Maze with location-dependent shocks and visualize it.
The data is recorded by Riddha Manna and Ana Marija Jaksic https://www.epfl.ch/labs/jaksic-lab.

"""
module YMaze

using Random, CSV, RollingFunctions, DataFrames, CategoricalArrays, Statistics
using DocStringExtensions, PrecompileTools

include("maze.jl")
include("encoders.jl")
include("io.jl")

# see PlottingExt
function plot_maze end
function plot_track end

# reduces TTFX
@setup_workload begin
    tmptrack = "track.csv"
    tmptime = "time.csv"
    tmppattern = "pattern.csv"
    CSV.write(joinpath(tempdir(), tmptrack),
              DataFrame(x = rand(1:10, 10), y = rand(1:10, 10)), header = false)
    CSV.write(joinpath(tempdir(), tmptime),
              DataFrame(t = rand(10)), header = false)
    CSV.write(joinpath(tempdir(), tmppattern),
              DataFrame(t = fill("GBB", 10)), header = false)
    @compile_workload begin
        read(tempdir(), tmptrack,
             load_arm = false, load_shock = false)
    end
    rm(joinpath(tempdir(), tmptrack))
    rm(joinpath(tempdir(), tmptime))
    rm(joinpath(tempdir(), tmppattern))
end
end # module
