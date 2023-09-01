const sqrt3 = sqrt(3)

function rotate(p, p₀, α)
    [cos(α) -sin(α)
     sin(α) cos(α)] * (p - p₀) + p₀
end
in_middle_turn(x, y) = (x - 100)^2 + (y - 20)^2 ≤ 16.25^2
in_right_turn(x, y) = (x - 177)^2 + (y - 153.5)^2 ≤ 16.25^2
in_left_turn(x, y) = (x - 22)^2 + (y - 153.5)^2 ≤ 16.25^2
in_middle_beam(x, y) = x ≥ 91.75 && x ≤ 108.25
# (110-8/sin(π/3))*sqrt(3)-100 to get ≈ 74
in_right_beam(x, y) = -x + sqrt3 * y ≥ 74 && -x + sqrt3 * y ≤ 107
in_left_beam(x, y) = x + sqrt3 * y ≥ 274 && x + sqrt3 * y ≤ 307

in_turn(x, y) = in_middle_turn(x, y) || in_left_turn(x, y) || in_right_turn(x, y)

const CENTER_POINTS = [[[100, y] for y in 20:5:110];
                       rotate.([[100, y] for y in 20:5:110], Ref([100, 110]), 120/180*π);
                       rotate.([[100, y] for y in 20:5:110], Ref([100, 110]), -120/180*π)]

function in_center(x, y)
    (in_middle_beam(x, y) && y > 88 && y < 110) ||
    (in_right_beam(x, y) && x < 125 && x > 100) ||
    (in_left_beam(x, y) && x > 75 && x ≤ 100)
end

function in_middle_arm(x, y)
    !in_center(x, y) && !in_middle_turn(x, y) && in_middle_beam(x, y) && y > 30 && y ≤ 88
end
function in_right_arm(x, y)
    !in_right_turn(x, y) &&
    in_right_beam(x, y) &&
    x < 180 &&
    x ≥ 125
end
function in_left_arm(x, y)
    !in_center(x, y) &&
    !in_left_turn(x, y) &&
    in_left_beam(x, y) &&
    x > 30 &&
    x ≤ 75
end

in_left(x, y) = in_left_arm(x, y) || in_left_turn(x, y)
in_right(x, y) = in_right_arm(x, y) || in_right_turn(x, y)
in_middle(x, y) = in_middle_arm(x, y) || in_middle_turn(x, y)

in_arm(x, y) = in_middle_arm(x, y) || in_left_arm(x, y) || in_right_arm(x, y)

function in_maze(x, y)
    in_middle_arm(x, y) ||
    in_left_arm(x, y) ||
    in_right_arm(x, y) ||
    in_middle_turn(x, y) ||
    in_left_turn(x, y) ||
    in_right_turn(x, y) ||
    in_center(x, y)
end

for f in (
    :in_center,
    :in_middle_turn,
    :in_right_turn,
    :in_left_turn,
    :in_turn,
    :in_middle_arm,
    :in_right_arm,
    :in_left_arm,
    :in_left,
    :in_right,
    :in_middle,
    :in_arm,
    :in_maze,
)
    direct_rule_key = string(f)[4:end]
    eval(quote
        $f(xy) = $f(xy[1], xy[2])
        function $f(x::ComponentVector)
            hasproperty(x, :x) && return $f(x.x, x.y)
            if hasproperty(x, Symbol($direct_rule_key))
                return getproperty(x, Symbol($direct_rule_key)) == 1
            end
            for k in subrule_keys($f)
                if hasproperty(x, k)
                    return getproperty(x, k) == 1
                end
            end
            false
        end
    end)
end
function subrule_keys(k)
    k == in_middle && return (:middle_arm, :middle_turn,)
    k == in_left && return (:left_arm, :left_turn)
    k == in_right && return (:right_arm, :right_turn)
    k == in_turn && return (:right_turn, :left_turn, :middle_turn)
    k == in_arm && return (:right_arm, :left_arm, :middle_arm)
    k == in_maze && return (sub_rule(:in_arm)..., subrule(:in_turn)..., :in_center)
    return tuple()
end
function in_shock_arm(x::ComponentVector)
    x.shock == 1 && return true
    hasproperty(x, :shockˌ1) && x.shockˌ1 == 1 && return true
    false
end
in_shock_arm(x, y) = false # this is a hack for rand_track in simulators.jl

function random_maze_position(; N = 1, vmax = 5)
    xs = Int[]
    ys = Int[]
    while true
        x = rand(1:200)
        y = rand(1:175)
        if in_maze(x, y)
            push!(xs, x)
            push!(ys, y)
            break
        end
    end
    while length(xs) < N
        x = xs[end] + rand(-vmax:vmax)
        y = ys[end] + rand(-vmax:vmax)
        if in_maze(x, y)
            push!(xs, x)
            push!(ys, y)
        end
    end
    (x = xs, y = ys)
end
