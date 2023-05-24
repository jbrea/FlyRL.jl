# API
"""
$SIGNATURES

Encode `track` with `encoder`.
"""
encode(encoder, track::DataFrame)
"""
$SIGNATURES

Encode `track` with `encoder` and store result in data frame `track`.
"""
encode!(encoder, track::DataFrame)

# For internal use:
# encode!(encoder, out::ComponentArray, in::ComponentArray)
# decode(decoder, data) (only for VectorEncoder)
# labels(encoder)
# levels(encoder)
#
# The current API is probably not yet ideal.
# - some encoders currently "bubble up" compressed_stream_idxs. Can this be done better?

abstract type AbstractEncoder end

function encode!(e::AbstractEncoder, track::DataFrame)
    for (k, v) in pairs(encode(e, track))
        setproperty!(track, k, v)
    end
    track
end
hascols(e::AbstractEncoder, track::DataFrame) = all(hasproperty.(Ref(track), labels(e)))
function getcols(e::AbstractEncoder, track::DataFrame)
    l = labels(e)
    NamedTuple{l}(tuple(getproperty.(Ref(track), l)...))
end

function distance2wall(
    x,
    y,
    ox,
    oy;
    stepsize = 0.5,
    max_distance = 30.0,
    scale = max_distance,
)
    for γ = 0:stepsize:max_distance
        in_maze(x + γ * ox, y + γ * oy) || return γ / scale
    end
    return max_distance / scale
end
function distance2wall(e::AbstractEncoder, x, y, ox, oy)
    distance2wall(
        x,
        y,
        ox,
        oy;
        stepsize = e.stepsize,
        max_distance = e.max_distance,
        scale = e.scale,
    )
end
struct Distance2WallsEncoder{E} <: AbstractEncoder
    max_distance::Float64
    scale::Float64
    stepsize::Float64
end
function Distance2WallsEncoder{E}(; max_distance = 40, scale = max_distance, stepsize = 1.5) where E
    Distance2WallsEncoder{E}(max_distance, scale, stepsize)
end
Distance2WallsEncoder(; kwargs...) = EightWallsEncoder(; kwargs...)
"""
    FourWallsEncoder(; max_distance = 40, scale = max_distance, stepsize = 1.5)

Encodes (x, y, ox, oy)-position-orientation tuples to relative distances to the next wall
at 0, 90, 180 and 270 degrees relative to the orientation.

If the distance of the wall is `max_distance` and `scale == max_distance` the returned
value is 1. The stepsize (in pixels) controls the granularity at which the distance to the
wall is computed.
"""
FourWallsEncoder(; kwargs...) = Distance2WallsEncoder{4}(; kwargs...)
"""
    SixWallsEncoder(; max_distance = 40, scale = max_distance, stepsize = 1.5)

Encodes (x, y, ox, oy)-position-orientation tuples to relative distances to the next wall
at 0, 45, 135, 180, 225 and 315 degrees relative to the orientation.

If the distance of the wall is `max_distance` and `scale == max_distance` the returned
value is 1. The stepsize (in pixels) controls the granularity at which the distance to the
wall is computed.
"""
SixWallsEncoder(; kwargs...) = Distance2WallsEncoder{6}(; kwargs...)
"""
    EightWallsEncoder(; max_distance = 40, scale = max_distance, stepsize = 1.5)

Encodes (x, y, ox, oy)-position-orientation tuples to relative distances to the next wall
at 0, 45, 90, 135, 180, 225, 270 and 315 degrees relative to the orientation.

If the distance of the wall is `max_distance` and `scale == max_distance` the returned
value is 1. The stepsize (in pixels) controls the granularity at which the distance to the
wall is computed.
"""
EightWallsEncoder(; kwargs...) = Distance2WallsEncoder{8}(; kwargs...)
labels(::Distance2WallsEncoder{4}) = (:ahead, :right, :left, :behind)
labels(::Distance2WallsEncoder{6}) = (:ahead, :right_a, :right_b, :left_a, :left_b, :behind)
labels(::Distance2WallsEncoder{8}) = (:ahead, :right_a, :right, :right_b, :left_a, :left, :left_b, :behind)
function _vecnt2ntvec(data)
    NamedTuple{keys(data[1])}(tuple(([data[j][i] for j in eachindex(data)] for i in eachindex(data[1]))...))
end
function encode(e::Distance2WallsEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    ox, oy = encode(OrientationEncoder(), track)
    _vecnt2ntvec(distance2walls.(Ref(e), track.x, track.y, ox, oy))
end
function distance2walls(e::Distance2WallsEncoder{E}, x, y, ox, oy) where E
    ahead = distance2wall(e, x, y, ox, oy)
    behind = distance2wall(e, x, y, -ox, -oy)
    if E == 8 || E == 4
        right = distance2wall(e, x, y, oy, -ox)
        left = distance2wall(e, x, y, -oy, ox)
    end
    if E == 8 || E == 6
        right_a = distance2wall(e, x, y, (ox + sqrt3*oy)/2, (-sqrt3*ox + oy)/2)
        right_b = distance2wall(e, x, y, (-ox + sqrt3*oy)/2, (-sqrt3*ox - oy)/2)
        left_a = distance2wall(e, x, y, (ox - sqrt3*oy)/2, (sqrt3*ox + oy)/2)
        left_b = distance2wall(e, x, y, (-ox - sqrt3*oy)/2, (sqrt3*ox - oy)/2)
    end
    if E == 4
        (; ahead, right, left, behind)
    elseif E == 6
        (; ahead, right_a, right_b, left_a, left_b, behind)
    elseif E == 8
        (; ahead, right_a, right, right_b, left_a, left, left_b, behind)
    end
end
@inline function encode!(e::Distance2WallsEncoder{E}, out::ComponentArray, in::ComponentArray) where E
    encode!(OrientationEncoder(), in, in)
    res = distance2walls(e, in.x, in.y, in.ox, in.oy)
    if E == 4
        out.ahead, out.right, out.left, out.behind = res
    elseif E == 6
        out.ahead, out.right_a, out.right_b, out.left_a, out.left_b, out.behind = res
    elseif E == 8
        out.ahead, out.right_a, out.right, out.right_b, out.left_a, out.left, out.left_b, out.behind = res
    end
    out
end

"""
    ClosestWallEncoder(encoder)

Encodes the closest wall as determined with `encoder` (either [`FourWallsEncoder`](@ref),
[`SixWallsEncoder`](@ref) or [`EightWallsEncoder`](@ref)).
"""
struct ClosestWallEncoder{E} <: AbstractEncoder
    encoder::E
end
ClosestWallEncoder() = ClosestWallEncoder(EightWallsEncoder())
labels(::ClosestWallEncoder) = (:closest_wall,)
function encode(e::ClosestWallEncoder, track)
    hascols(e, track) && return getcols(e, track)
    data = encode(e.encoder, track) |> DataFrame
    (; closest_wall = minimum.(eachrow(data)))
end
function encode!(e::ClosestWallEncoder, out::ComponentArray, in::ComponentArray)
    encode!(OrientationEncoder(), in, in)
    out.closest_wall = minimum(distance2walls(e.encoder, in.x, in.y, in.ox, in.oy))
    out
end

"""
    OrientationEncoder()

Encodes orientations based on Δx and Δy.
"""
struct OrientationEncoder <: AbstractEncoder end
encode_dim(::OrientationEncoder) = 2
function first_orientation(Δx, Δy)
    for i in eachindex(Δx)
        if !ismissing(Δx[i]) && (Δx[i] != 0 || Δy[i] != 0)
            n = sqrt(Δx[i]^2 + Δy[i]^2)
            return Δx[i] / n, Δy[i] / n
        end
    end
    return 1.0, 0.0
end
function encode(e::OrientationEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    Δx, Δy = encode(DeltaPositionEncoder(), track)
    encode(e, Δx, Δy)
end
function encode(::OrientationEncoder, Δx, Δy)
    T = length(Δx)
    ox = zeros(T)
    oy = zeros(T)
    ox[1], oy[1] = first_orientation(Δx, Δy)
    for i = 2:T
        ox[i], oy[i] = orientation(Δx[i], Δy[i], speed(Δx[i], Δy[i]), ox[i-1], oy[i-1])
    end
    (; ox, oy)
end
function orientation(Δx, Δy, speed, old_ox, old_oy)
    if !ismissing(speed) && speed > 0
        Δx/speed, Δy/speed
    else
        old_ox, old_oy
    end
end
@inline function encode!(::OrientationEncoder, out::ComponentArray, in::ComponentArray)
    dx = in.Δx
    dy = in.Δy
    out.ox, out.oy = orientation(dx, dy, speed(dx, dy), in.ox, in.oy)
    out
end
labels(::OrientationEncoder) = (:ox, :oy)

"""
    DeltaPositionIndexEncoder(; maxradius = 20, stepsize = 1)

Encodes relative positions (Δx, Δy) with an integer, based on a 2D grid with `stepsize`
and largest relative position `Δx^2 + Δy^2 = maxradius^2`.
"""
struct DeltaPositionIndexEncoder <: AbstractEncoder
    lookup::Dict{Tuple{Int,Int}, Int}
    dx::Vector{Int}
    dy::Vector{Int}
end
function DeltaPositionIndexEncoder(; kwargs...)
    dx, dy = delta_position_grid(; kwargs...)
    DeltaPositionIndexEncoder(Dict((dxi, dyi) => i for (i, (dxi, dyi)) in enumerate(zip(dx, dy))), dx, dy)
end
labels(::DeltaPositionIndexEncoder) = (:delta_position_index,)
function encode(e::DeltaPositionIndexEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    Δx, Δy = encode(DeltaPositionEncoder(), track)
    encode(e, Δx, Δy)
end
function encode(e::DeltaPositionIndexEncoder, Δx, Δy)
    (; delta_position_index = [get(e.lookup, (Δxi, Δyi), missing) for (Δxi, Δyi) in zip(Δx, Δy)])
end



levels(e) = ntuple(_ -> Number, length(labels(e)))

struct ArmEncoder <: AbstractEncoder
    levels::Vector{String}
end
"""
$SIGNATURES

Encodes arms of the maze as `"left"`, `"middle"`, `"right"`, `"center"`.
Points outside the maze are marked as `"outlier"` if `with_outliers = true`.
"""
function ArmEncoder(; with_outliers = false)
    levels = FlyRL.with_outliers(["left", "middle", "right", "center"], with_outliers)
    ArmEncoder(levels)
end
with_outliers(x, with_outliers; outlier = "outlier") = ifelse(with_outliers, [x; outlier], x)
levels(e::ArmEncoder) = (e.levels,)
labels(::ArmEncoder) = (:arm,)
function encode(e::ArmEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    (;
        arm = categorical(
            encode.(Ref(e), track.x, track.y),
            levels = levels(e)[1],
        )
    )
end
function encode(::ArmEncoder, x::Number, y::Number)
    (in_left_arm(x, y) || in_left_turn(x, y)) && return "left"
    (in_middle_arm(x, y) || in_middle_turn(x, y)) && return "middle"
    (in_right_arm(x, y) || in_right_turn(x, y)) && return "right"
    in_center(x, y) && return "center"
    return "outlier"
end

struct ShockArmEncoder <: AbstractEncoder
    levels::Vector{String}
end
"""
$SIGNATURES

Encodes arms of the maze as `"neutral A"`, `"neutral B"`, `"shock"` and `"center"`.
Points outside the maze are marked as `"outlier"` if `with_outliers = true`.
"""
function ShockArmEncoder(; with_outliers = false)
    levels = FlyRL.with_outliers(["neutral A", "neutral B", "shock", "center"],
                                 with_outliers)
    ShockArmEncoder(levels)
end
levels(e::ShockArmEncoder) = (e.levels,)
labels(::ShockArmEncoder) = (:shock_arm,)
function encode(e::ShockArmEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    (;
        shock_arm = categorical(
            encode.(Ref(e), track.x, track.y, track.pattern),
            levels = levels(e)[1],
        )
    )
end
function encode(::ShockArmEncoder, x::Number, y::Number, pattern)
    (in_left_arm(x, y) || in_left_turn(x, y)) && return pattern[1] == 'G' ? "shock" : "neutral A"
    (in_middle_arm(x, y) || in_middle_turn(x, y)) && return pattern[2] == 'G' ? "shock" : pattern[1] == 'G' ? "neutral A" : "neutral B"
    (in_right_arm(x, y) || in_right_turn(x, y)) && return pattern[3] == 'G' ? "shock" : "neutral B"
    in_center(x, y) && return "center"
    return "outlier"
end

struct SemanticEncoder3 <: AbstractEncoder
    levels::Vector{String}
end
"""
$SIGNATURES

Encodes positions of the maze as `"center"`, `"arm"` or `"turn"`.
Points outside the maze are marked as `"outlier"` if `with_outliers = true`.
"""
function SemanticEncoder3(; with_outliers = false)
    levels = FlyRL.with_outliers(["center", "arm", "turn"], with_outliers)
    SemanticEncoder3(levels)
end
levels(e::SemanticEncoder3) = (e.levels,)
labels(::SemanticEncoder3) = (:se3,)
function encode(e::SemanticEncoder3, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    (;
        se3 = categorical(
            encode.(Ref(e), track.x, track.y),
            levels = levels(e)[1],
        )
    )
end
function encode(::SemanticEncoder3, x::Number, y::Number)
    in_center(x, y) && return "center"
    in_arm(x, y) && return "arm"
    in_turn(x, y) && return "turn"
    return "outlier"
end
function encode!(::SemanticEncoder3, out::ComponentArray, in::ComponentArray)
    out.center = in_center(in.x, in.y)
    out.arm = in_arm(in.x, in.y)
    out.turn = in_turn(in.x, in.y)
    out
end

struct SemanticEncoder7 <: AbstractEncoder
    levels::Vector{String}
end
"""
$SIGNATURES

Encodes positions of the maze as `"center"`, `"left arm"`, `"left turn"`, `"middle arm"`, `"middle turn"`, `"right arm"`, `"right turn"`.
Points outside the maze are marked as `"outlier"` if `with_outliers = true`.
"""
function SemanticEncoder7(; with_outliers = false)
    levels = FlyRL.with_outliers(["center", "left arm", "left turn", "middle arm", "middle turn", "right arm", "right turn"], with_outliers)
    SemanticEncoder7(levels)
end
levels(e::SemanticEncoder7) = (e.levels,)
labels(::SemanticEncoder7) = (:se7,)
function encode(e::SemanticEncoder7, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    (;
        se7 = categorical(
            encode.(Ref(e), track.x, track.y),
            levels = levels(e)[1],
        )
    )
end
function encode(::SemanticEncoder7, x::Number, y::Number)
    in_center(x, y) && return "center"
    in_left_arm(x, y) && return "left arm"
    in_left_turn(x, y) && return "left turn"
    in_middle_arm(x, y) && return "middle arm"
    in_middle_turn(x, y) && return "middle turn"
    in_right_arm(x, y) && return "right arm"
    in_right_turn(x, y) && return "right turn"
    return "outlier"
end

"""
    VelocityEncoder()

Encodes velocities based on Δx, Δy and Δt.
"""
struct VelocityEncoder <: AbstractEncoder end
function encode(e::VelocityEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    Δt, = encode(DeltaTimeEncoder(), track)
    Δx, Δy = encode(DeltaPositionEncoder(), track)
    encode(e, Δx, Δy, Δt)
end
function encode(::VelocityEncoder, Δx, Δy, Δt)
    (; vx = Δx ./ Δt, vy = Δy ./ Δt)
end
labels(::VelocityEncoder) = (:vx, :vy)

const DELTA_CONVENTION_DOCSTRING = """
Convention: the same row should contain the delta values that
lead to the current value, i.e. delta values should start with `missing`.
"""
"""
    DeltaPositionEncoder()

Encodes `Δx` and `Δy` from `x` and `y`.

$DELTA_CONVENTION_DOCSTRING
"""
struct DeltaPositionEncoder <: AbstractEncoder end
function encode(e::DeltaPositionEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    encode(e, track.x, track.y)
end
function encode(::DeltaPositionEncoder, x, y)
    T = length(x)
    Δx = Array{Union{Float64,Missing}}(undef, T)
    Δy = Array{Union{Float64,Missing}}(undef, T)
    for i = 2:T
        Δx[i] = x[i] - x[i-1]
        Δy[i] = y[i] - y[i-1]
    end
    (; Δx, Δy)
end
labels(::DeltaPositionEncoder) = (:Δx, :Δy)

"""
    DeltaTimeEncoder()

Encodes `Δt` from `t`; uses default $DEFAULT_Δt, if `t` is missing.

$DELTA_CONVENTION_DOCSTRING
"""
Base.@kwdef struct DeltaTimeEncoder <: AbstractEncoder
    default_Δt::Float64 = DEFAULT_Δt
end
labels(::DeltaTimeEncoder) = (:Δt,)
function encode(e::DeltaTimeEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    if hasproperty(track, :t)
        Δt = [missing; track.t[2:end] - track.t[1:end-1]]
    else
        Δt = fill(e.default_Δt, nrow(track))
    end
    (; Δt)
end
"""
    FutureDeltaTimeEncoder()

Same as [`DeltaTimeEncoder`]() but offset by one, i.e. the same row contains the delta
value that leads to the next value.
"""
Base.@kwdef struct FutureDeltaTimeEncoder <: AbstractEncoder
    default_Δt::Float64 = DEFAULT_Δt
end
labels(::FutureDeltaTimeEncoder) = (:futureΔt,)
function encode(e::FutureDeltaTimeEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    Δt, = encode(DeltaTimeEncoder(), track)
    (; futureΔt = [Δt; missing])
end

struct SpeedEncoder{R} <: AbstractEncoder
    outlier_threshold::Float64
end
"""
$SIGNATURES

Encode the speed based on `vx` and `vy` (see [`VelocityEncoder`](@ref)).
Speed values above `outlier_threshold` are returned as `missing`.
If `return_outliers = true`, return also a vector of outlier values.
"""
function SpeedEncoder(; outlier_threshold = 160, return_outliers = false)
    SpeedEncoder{return_outliers}(outlier_threshold)
end
function encode(e::SpeedEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    vx, vy = encode(VelocityEncoder(), track)
    encode(e, vx, vy)
end
labels(::SpeedEncoder{true}) = (:speed, :speed_outlier)
labels(::SpeedEncoder{false}) = (:speed,)
function encode(e::SpeedEncoder{return_outliers}, vx, vy) where return_outliers
    T = length(vx)
    speed = Array{Union{Float64,Missing}}(undef, T)
    if return_outliers
        speed_outlier = Array{Union{Float64,Missing}}(undef, T)
    end
    for i = 1:T
        n = FlyRL.speed(vx[i], vy[i])
        if !ismissing(n) && n ≤ e.outlier_threshold
            speed[i] = n
        elseif return_outliers
            speed_outlier[i] = n
        end
    end
    if return_outliers
        (; speed, speed_outlier)
    else
        (; speed)
    end
end
function speed(vx, vy)
    (ismissing(vx) || ismissing(vy)) && return missing
    vx == vy == zero(vx) && return zero(vx)
    sqrt(vx^2 + vy^2)
end
function encode!(::SpeedEncoder, out::ComponentArray, in::ComponentArray)
    Δt = if hasproperty(in, :Δt)
        in.Δt
    else
        DEFAULT_Δt
    end
    vx, vy = encode(VelocityEncoder(), in.Δx, in.Δy, Δt)
    out.speed = speed(vx, vy) # doesn't care about outliers
end

"""
    AngleEncoder()

Encodes angles between subsequent orientations.

$DELTA_CONVENTION_DOCSTRING
"""
struct AngleEncoder <: AbstractEncoder end
angle(x, y) = angle(x[1], x[2], y[1], y[2])
angle(ox, oy, old_ox, old_oy) = atan(ox * old_oy - oy * old_ox, ox * old_ox + oy * old_oy)
function encode(e::AngleEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    speed, = encode(SpeedEncoder(), track)
    ox, oy = encode(OrientationEncoder(), track)
    encode(e, speed, ox, oy)
end
function encode(::AngleEncoder, speed, ox, oy)
    T = length(ox)
    angle = Array{Union{Float64,Missing}}(undef, T)
    for i = 2:T
        angle[i] = FlyRL.angle(ox[i], oy[i], ox[i-1], oy[i-1], speed[i])
    end
    (; angle)
end
function angle(ox, oy, old_ox, old_oy, speed)
    (ismissing(speed) || speed == 0) && return 0.
    angle(ox, oy, old_ox, old_oy)
end
@inline function encode!(::AngleEncoder, out::ComponentArray, in::ComponentArray)
    encode!(OrientationEncoder(), in, in)
    encode!(SpeedEncoder(), in, in)
    out.angle = angle(in.ox, in.oy, in.oxˌ1, in.oyˌ1, in.speed)
    out
end
labels(::AngleEncoder) = (:angle,)
"""
    AngleEncoder2D()

Encodes angles as `(sin(angle), cos(angle))` (see also [`AngleEncoder`](@ref)).
"""
struct AngleEncoder2D <: AbstractEncoder end
labels(::AngleEncoder2D) = (:sin_angle, :cos_angle)
function encode(e::AngleEncoder2D, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    angle = encode(AngleEncoder(), track).angle
    (sin_angle = sin.(angle), cos_angle = cos.(angle))
end
function encode!(::AngleEncoder2D, out::ComponentArray, in::ComponentArray)
    encode!(AngleEncoder(), in, in)
    out.sin_angle = sin(in.angle)
    out.cos_angle = cos(in.angle)
    out
end

"""
    ColumnPicker(colname::Symbol)

Picks a column from a data frame.
"""
struct ColumnPicker <: AbstractEncoder
    colname::Symbol
end
function encode(e::ColumnPicker, track::DataFrame)
    (; e.colname => getproperty(track, e.colname))
end
@inline function encode!(e::ColumnPicker, out::ComponentArray, in::ComponentArray)
    setproperty!(out, e.colname, getproperty(in, e.colname))
    out
end
labels(e::ColumnPicker) = (e.colname,)

"""
    InShockArmEncoder()

Encodes if the position is in the shock arm.
"""
struct InShockArmEncoder <: AbstractEncoder end
function encode(e::InShockArmEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    (; inshockarm = [shock_function(pat)(x, y)
                     for (pat, x, y) in zip(track.pattern, track.x, track.y)])
end
labels(::InShockArmEncoder) = (:inshockarm,)

struct ColorEncoder <: AbstractEncoder
    colordict::Dict{Char,String}
    levels::Vector{String}
end
"""
$SIGNATURES

Encode color based on `track.pattern`.
"""
function ColorEncoder(; colordict = Dict('B' => "blue", 'G' => "green", 'R' => "red"),
                        with_outliers = false)
    l = FlyRL.with_outliers(["gray"; values(colordict)...], with_outliers; outlier = "black")
    ColorEncoder(colordict, l)
end
labels(::ColorEncoder) = (:color,)
levels(e::ColorEncoder) = (e.levels,)
function color(pattern, x, y, colordict = Dict('B' => "blue", 'G' => "green", 'R' => "red"))
    in_center(x, y) && return "gray"
    in_left(x, y) && return colordict[pattern[1]]
    in_middle(x, y) && return colordict[pattern[2]]
    in_right(x, y) && return colordict[pattern[3]]
    return "black"
end
function encode(e::ColorEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    (;
        color = categorical(
            color.(track.pattern, track.x, track.y, Ref(e.colordict)),
            levels = FlyRL.levels(e)[1],
        )
    )
end

"""
    LevelEncoder(encoder)

Encode level as an integer for a categorical encoder (like [`ShockArmEncoder`](@ref)).
"""
struct LevelEncoder{E} <: AbstractEncoder
    encoder::E
end
labels(e::LevelEncoder) = (Symbol("$(labels(e.encoder)[1])_level"),)
levels(e::LevelEncoder) = levels(e.encoder)
function encode(e::LevelEncoder, track::DataFrame)
    hascols(e, track) && return getcols(e, track)
    NamedTuple{labels(e)}((levelcode.(first(encode(e.encoder, track))),))
end

struct VectorEncoder{T, E, K} <: AbstractEncoder
    encoders::E
    dropmissing::Bool
    with_intercept::Bool
    keys::K
end
markov(e::VectorEncoder) = maximum(markov.(e.encoders))
isdynamic(e::VectorEncoder) = any(isdynamic.(e.encoders))
labels(::VectorEncoder) = (:vectors,)
function _keys(e::Tuple, with_intercept)
    lb = tuple(Iterators.flatten(labels.(e))...)
    le = tuple(Iterators.flatten(levels.(e))...)
    ks = vcat([lei == Number ? lbi : add_suffix.(Symbol.(lei), Ref(suffix(lbi))) for (lbi, lei) in zip(lb, le)]...)
    if with_intercept
        push!(ks, :intercept)
    end
    tuple(ks...)
end
"""
$SIGNATURES

Encode multiple encodings in a single vector.
Categorical encoders are one-hot encoded.

## Example
```
encode(VectorEncoder(ShockArmEncoder(), SpeedEncoder()), random_track(N = 10)) |> DataFrame
```
"""
function VectorEncoder(encoders...; T = Float64, dropmissing = false, intercept = false)
    ks = _keys(encoders, intercept)
    VectorEncoder{T, typeof(encoders), typeof(ks)}(encoders, dropmissing, intercept, ks)
end
function mysetindex!(x, v::Number, o)
    o[] += 1
    setindex!(x, v, o[])
    x
end
function mysetindex!(x, v::Union{Tuple,AbstractVector{<:Any}}, o)
    for i in eachindex(v)
        o[] += 1
        setindex!(x, v[i], o[])
    end
    x
end
function mysetindex!(x, v::CategoricalVector, o)
    for vi in v
        mysetindex!(x, vi, o)
    end
    x
end
function mysetindex!(x, v::CategoricalValue, o)
    for level in CategoricalArrays.levels(v)
        o[] += 1
        if v == level
            setindex!(x, 1.0, o[])
        else
            setindex!(x, 0.0, o[])
        end
    end
    x
end
drop(nt::NamedTuple, k...) = Base.structdiff(nt, NamedTuple{k})
drop(t::Tuple, k...) = filter(∉(k), t)
function bubble_up_idxs(encoders, track)
    data = merge(encode.(encoders, Ref(track))...)
    compressed_stream_idxs = nothing
    if hasproperty(data, :compressed_stream_idxs)
        compressed_stream_idxs = data.compressed_stream_idxs
        data = drop(data, :compressed_stream_idxs)
    end
    data, compressed_stream_idxs
end
function encode(e::VectorEncoder{T}, track::DataFrame) where T
    data, compressed_stream_idxs = bubble_up_idxs(e.encoders, track)
    N = length(first(data))
    M = length(e.keys)
    vectors = [[ComponentVector(NamedTuple{e.keys}(zeros(T, M))) for _ = 1:N]; missing]
    k = 1
    o = Base.RefValue{Int}(0)
    idxs = Int[]
    for i = 1:N
        if e.with_intercept
            vectors[k].intercept = 1
        end
        o[] = 0
        skip = false
        for v in data
            if any(ismissing.(v[i]))
                if e.dropmissing
                    skip = true
                else
                    vectors[k] = missing
                end
                break
            end
            mysetindex!(vectors[k], v[i], o)
        end
        skip && continue
        push!(idxs, i)
        k += 1
    end
    if isnothing(compressed_stream_idxs)
        (; vectors = vectors[1:(k - 1)])
    else
        (; vectors = vectors[1:(k - 1)], compressed_stream_idxs = compressed_stream_idxs[idxs])
    end
end
@inline function encode!(e::VectorEncoder, out::ComponentArray, in::ComponentArray)
    encode!(e.encoders, out, in)
    out
end
@inline function encode!(e::Tuple, out::ComponentArray, in::ComponentArray)
    encode!(first(e), out, in)
    encode!(Base.tail(e), out, in)
    out
end
@inline encode!(::Tuple{}, ::ComponentArray, ::ComponentArray) = nothing
_convert(v, o, ::Type{Number}) = v[o]
_convert(v, o, levels) = levels[Bool.(v[o:o+length(levels)-1])][]
"""
$SIGNATURES

Inverse of `encode(VectorEncoder(), track)`.
"""
function decode(e::VectorEncoder, vectors)
    lb = tuple(Iterators.flatten(labels.(e.encoders))...)
    le = tuple(Iterators.flatten(levels.(e.encoders))...)
    data = []
    o = 1
    for lei in le
        if lei == Number
            push!(data, [v[o] for v in vectors])
            o += 1
        else
            push!(data, categorical([lei[Bool.(v[o:o+length(lei)-1])][] for v in vectors], levels = lei))
            o += length(lei)
        end
    end
    NamedTuple{lb}(tuple(data...))
end

struct MarkovKEncoder{K,E} <: AbstractEncoder
    encoders::E
end
markov(::Any) = 1
markov(::MarkovKEncoder{K}) where K = K
isdynamic(e::MarkovKEncoder) = any(isdynamic.(e.encoders))
"""
$SIGNATURES

Create copies of the original encoders with offsets up to `K`.
Appends `ˌk` to copy with offset `k`.
## Example
```
encode(MarkovKEncoder(3, DynamicCompressEncoder(:arm, ArmEncoder())), random_track()) |> DataFrame
```
"""
MarkovKEncoder(K, encoders...) = MarkovKEncoder{K, typeof(encoders)}(encoders)
add_suffix(s, i) = Symbol("$s" * (i == 0 ? "" : "ˌ$i")) # \verti
split_suffix(s) = split("$s", 'ˌ') # \verti
function suffix(s)
    x = split_suffix(s)
    length(x) == 2 && return Meta.parse(x[2])
    0
end
function labels(e::MarkovKEncoder{K}) where K
    l = tuple(Iterators.flatten(labels.(e.encoders))...)
    if K > 1
        tuple(vcat([vcat(map(i -> add_suffix(li, i), 0:K-1)...) for li in l]...)...)
    else
        l
    end
end
function levels(e::MarkovKEncoder{K}) where K
    tuple(vcat([fill(l, K) for l in Iterators.flatten(levels.(e.encoders))]...)...)
end
function encode(e::MarkovKEncoder{K}, track::DataFrame) where K
    hascols(e, track) && return getcols(e, track)
    data, compressed_stream_idxs = bubble_up_idxs(e.encoders, track)
    l = labels(e)
    newdata = vcat([[[fill(missing, k-1); d[1:end-k+1]] for k in 1:K] for d in data]...)
    if !isnothing(compressed_stream_idxs)
        l = tuple(l..., :compressed_stream_idxs)
        newdata = [newdata..., compressed_stream_idxs]
    end
    NamedTuple{l}(newdata)
end
# This is a hack!
function encode!(::MarkovKEncoder{2,<:Tuple{<:AngleEncoder2D}}, out::ComponentArray, in::ComponentArray)
    out.sin_angleˌ1 = sin(in.angleˌ1)
    out.cos_angleˌ1 = cos(in.angleˌ1)
    encode!(AngleEncoder2D(), out, in)
    out
end
function encode!(::MarkovKEncoder{2,<:Tuple{<:SpeedEncoder}}, out::ComponentArray, in::ComponentArray)
    out.speedˌ1 = in.speedˌ1
    encode!(SpeedEncoder(), out, in)
    out
end

"""
    DurationPerStateEncoder(encoder)

Encode the duration that is spent in a given state.
## Example
encode(DurationPerStateEncoder(ShockArmEncoder()), random_track()) |> DataFrame
"""
struct DurationPerStateEncoder{E} <: AbstractEncoder
    encoder::E
end
function labels(e::DurationPerStateEncoder)
    l = labels(e.encoder)
    length(l) > 1 && error("Expecting encoder that returns only one column. Got encoder $(e.encoder) which returns columns $l.")
    (l[1], Symbol(l[1], "_Δt"))
end
function encode(e::DurationPerStateEncoder, track)
    data, idxs = encode(DynamicCompressEncoder(labels(e.encoder)[1], e.encoder), track)
    push!(idxs, nrow(track))
    NamedTuple{labels(e)}((data, track.t[idxs[2:end]] - track.t[idxs[1:end-1]]))
end

struct DynamicCompressEncoder{E,K} <: AbstractEncoder
    encoders::E
    compress_on::K
end
"""
    DynamicCompressEncoder(compress_on, encoders...)

Merge subsequent states of `compress_on` into one state.
## Example
encode(DynamicCompressEncoder(ShockArmEncoder()), random_track()) |> DataFrame
"""
function DynamicCompressEncoder(compress_on::Union{Symbol,NTuple{N,Symbol}}, e...) where N
    if isa(compress_on, Symbol)
        compress_on = tuple(compress_on)
    end
    DynamicCompressEncoder{typeof(e), typeof(compress_on)}(e, compress_on)
end
markov(e::DynamicCompressEncoder) = maximum(markov.(e.encoders))
isdynamic(::Any) = false
isdynamic(::DynamicCompressEncoder) = true
function labels(e::DynamicCompressEncoder)
    tuple(Iterators.flatten(labels.(e.encoders))...)
end
function levels(e::DynamicCompressEncoder)
    tuple(Iterators.flatten(levels.(e.encoders))...)
end
function encode(e::DynamicCompressEncoder, track::DataFrame)
    data = merge(encode.(e.encoders, Ref(track))...)
    compressed_stream_idxs = [1]
    colstocheck = getproperty.(Ref(data), e.compress_on)
    state = first.(colstocheck)
    for i = 2:nrow(track)
        newstate = getindex.(colstocheck, i)
        if newstate != state
            state = newstate
            push!(compressed_stream_idxs, i)
        end
    end
    ks = tuple(labels(e)..., :compressed_stream_idxs)
    NamedTuple{ks}(tuple(getindex.(values(data), Ref(compressed_stream_idxs))..., compressed_stream_idxs))
end
