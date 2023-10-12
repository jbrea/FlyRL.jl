# API
"""
$SIGNATURES

Encode `track` with `encoder`.
"""
encode(encoder, track)
"""
$SIGNATURES

Encode `track` with `encoder` and store result in data frame `track`.
"""
encode!(encoder, track)

abstract type AbstractEncoder end

function encode!(e::AbstractEncoder, track)
    for (k, v) in pairs(encode(e, track))
        setproperty!(track, k, v)
    end
    track
end
hascols(e::AbstractEncoder, track) = all(hasproperty.(Ref(track), labels(e)))
function getcols(e::AbstractEncoder, track)
    l = labels(e)
    NamedTuple{l}(tuple(getproperty.(Ref(track), l)...))
end

struct ArmEncoder <: AbstractEncoder
    levels::Vector{String}
end
"""
$SIGNATURES

Encodes arms of the maze as `"left"`, `"middle"`, `"right"`, `"center"`.
Points outside the maze are marked as `"outlier"` if `with_outliers = true`.
"""
function ArmEncoder(; with_outliers = false)
    levels = YMaze.with_outliers(["L", "M", "R", "X"], with_outliers)
    ArmEncoder(levels)
end
with_outliers(x, with_outliers; outlier = "outlier") = ifelse(with_outliers, [x; outlier], x)
levels(e::ArmEncoder) = (e.levels,)
labels(::ArmEncoder) = (:arm,)
function encode(e::ArmEncoder, track)
    hascols(e, track) && return getcols(e, track)
    (;
        arm = categorical(
            encode.(Ref(e), track.x, track.y),
            levels = levels(e)[1],
        )
    )
end
function encode(::ArmEncoder, x::Number, y::Number)
    (in_left_arm(x, y) || in_left_turn(x, y)) && return "L"
    (in_middle_arm(x, y) || in_middle_turn(x, y)) && return "M"
    (in_right_arm(x, y) || in_right_turn(x, y)) && return "R"
    in_center(x, y) && return "X"
    return "outlier"
end

struct ShockArmEncoder <: AbstractEncoder
    levels::Vector{String}
end
"""
$SIGNATURES

Encodes arms of the maze as `"neutral right"`, `"neutral left"`, `"shock"` and `"center"`.
Points outside the maze are marked as `"outlier"` if `with_outliers = true`.
"""
function ShockArmEncoder(; with_outliers = false)
    levels = YMaze.with_outliers(["neutral right", "neutral left", "shock", "center"],
                                 with_outliers)
    ShockArmEncoder(levels)
end
levels(e::ShockArmEncoder) = (e.levels,)
labels(::ShockArmEncoder) = (:state,)
function encode(e::ShockArmEncoder, track)
    hascols(e, track) && return getcols(e, track)
    (;
        shock_arm = categorical(
            encode.(Ref(e), track.x, track.y, track.pattern),
            levels = levels(e)[1],
        )
    )
end
function encode(::ShockArmEncoder, x::Number, y::Number, pattern)
    in_left(x, y) && return pattern[1] == 'G' ? "shock" :
                            pattern[2] == 'G' ? "neutral right" : "neutral left"
    in_middle(x, y) && return pattern[2] == 'G' ? "shock" :
                              pattern[1] == 'G' ? "neutral left" : "neutral right"
    in_right(x, y) && return pattern[3] == 'G' ? "shock" :
                             pattern[1] == 'G' ? "neutral right" : "neutral left"
    in_center(x, y) && return "center"
    return "outlier"
end

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
    l = YMaze.with_outliers(["gray"; values(colordict)...], with_outliers; outlier = "black")
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
function encode(e::ColorEncoder, track)
    hascols(e, track) && return getcols(e, track)
    (;
        color = categorical(
            color.(track.pattern, track.x, track.y, Ref(e.colordict)),
            levels = YMaze.levels(e)[1],
        )
    )
end

