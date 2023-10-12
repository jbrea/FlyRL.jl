const BOX_WIDTH = 200
const BOX_HEIGHT = ceil(Int, BOX_WIDTH * 23 / 26)
const CENTER_X = BOX_WIDTH ÷ 2
const CENTER_Y = ceil(Int, 15 * BOX_HEIGHT / 23)
const CENTER_LENGTH = 25
const TURN_RADIUS_SQUARED = 16^2
const MIDDLE_TURN_CENTER_Y = 26
const HALF_ARM_WIDTH = 8

function rotate(p, p₀, α)
    [cos(α) -sin(α)
     sin(α) cos(α)] * (p - p₀) + p₀
end
const SQRT3 = sqrt(3)
const CENTER = [CENTER_X, CENTER_Y]
const CENTER_POINTS = let points = [[CENTER_X, y] for y in MIDDLE_TURN_CENTER_Y:5:CENTER_Y]
    [points;
     rotate.(points, Ref(CENTER), 2/3*π);
     rotate.(points, Ref(CENTER), -2/3*π)]
end
const RIGHT_TURN_CENTER_X, RIGHT_TURN_CENTER_Y = rotate([CENTER_X, MIDDLE_TURN_CENTER_Y], CENTER, 2π/3)
const LEFT_TURN_CENTER_X, LEFT_TURN_CENTER_Y = rotate([CENTER_X, MIDDLE_TURN_CENTER_Y], CENTER, -2π/3)

right_beam(x, y) = -x + SQRT3 * y
right_beam_ortho(x, y) = SQRT3 * x + y
left_beam(x, y) = x + SQRT3 * y
left_beam_ortho(x, y) = -SQRT3 * x + y
const RIGHT_BEAM_LOWER = right_beam(rotate([CENTER_X-HALF_ARM_WIDTH, MIDDLE_TURN_CENTER_Y], CENTER, 2π/3)...)
const RIGHT_BEAM_UPPER = right_beam(rotate([CENTER_X+HALF_ARM_WIDTH, MIDDLE_TURN_CENTER_Y], CENTER, 2π/3)...)
const RIGHT_CENTER_BORDER = right_beam_ortho(rotate([CENTER_X, CENTER_Y - CENTER_LENGTH], CENTER, 2π/3)...)
const LEFT_BEAM_LOWER = left_beam(rotate([CENTER_X+HALF_ARM_WIDTH, MIDDLE_TURN_CENTER_Y], CENTER, -2π/3)...)
const LEFT_BEAM_UPPER = left_beam(rotate([CENTER_X-HALF_ARM_WIDTH, MIDDLE_TURN_CENTER_Y], CENTER, -2π/3)...)
const LEFT_CENTER_BORDER = left_beam_ortho(rotate([CENTER_X, CENTER_Y - CENTER_LENGTH], CENTER, -2π/3)...)

function in_middle_turn(x, y)
    (x - CENTER_X)^2 + (y - MIDDLE_TURN_CENTER_Y)^2 ≤ TURN_RADIUS_SQUARED
end
function in_right_turn(x, y)
    (x - RIGHT_TURN_CENTER_X)^2 + (y - RIGHT_TURN_CENTER_Y)^2 ≤ TURN_RADIUS_SQUARED
end
function in_left_turn(x, y)
    (x - LEFT_TURN_CENTER_X)^2 + (y - LEFT_TURN_CENTER_Y)^2 ≤ TURN_RADIUS_SQUARED
end
in_middle_beam(x, y) = CENTER_X - HALF_ARM_WIDTH ≤ x ≤ CENTER_X + HALF_ARM_WIDTH
in_right_beam(x, y) = RIGHT_BEAM_LOWER ≤ right_beam(x, y) ≤ RIGHT_BEAM_UPPER
in_left_beam(x, y) = LEFT_BEAM_LOWER ≤ left_beam(x, y) ≤ LEFT_BEAM_UPPER

in_turn(x, y) = in_middle_turn(x, y) || in_left_turn(x, y) || in_right_turn(x, y)

function in_center(x, y)
    (in_middle_beam(x, y) && y ≥ CENTER_Y - CENTER_LENGTH && y ≤ CENTER_Y) ||
    (in_right_beam(x, y) && right_beam_ortho(x, y) ≤ RIGHT_CENTER_BORDER && x ≥ CENTER_X) ||
    (in_left_beam(x, y) && left_beam_ortho(x, y) ≤ LEFT_CENTER_BORDER && x ≤ CENTER_X)
end

function in_middle_arm(x, y)
    !in_center(x, y) && !in_middle_turn(x, y) && in_middle_beam(x, y) && y > MIDDLE_TURN_CENTER_Y && y ≤ CENTER_Y
end
function in_right_arm(x, y)
    !in_center(x, y) &&
    !in_right_turn(x, y) &&
    in_right_beam(x, y) &&
    x < RIGHT_TURN_CENTER_X &&
    x ≥ CENTER_X
end
function in_left_arm(x, y)
    !in_center(x, y) &&
    !in_left_turn(x, y) &&
    in_left_beam(x, y) &&
    x > LEFT_TURN_CENTER_X &&
    x ≤ CENTER_X
end

in_left(x, y) = in_left_arm(x, y) || in_left_turn(x, y)
in_right(x, y) = in_right_arm(x, y) || in_right_turn(x, y)
in_middle(x, y) = in_middle_arm(x, y) || in_middle_turn(x, y)

in_arm(x, y) = in_middle_arm(x, y) || in_left_arm(x, y) || in_right_arm(x, y)

function in_maze(x, y)
    in_middle(x, y) ||
    in_left(x, y) ||
    in_right(x, y) ||
    in_center(x, y)
end


function random_maze_position(; N = 1, vmax = 5, rng = Random.default_rng())
    xs = Int[]
    ys = Int[]
    while true
        x = rand(rng, 1:200)
        y = rand(rng, 1:175)
        if in_maze(x, y)
            push!(xs, x)
            push!(ys, y)
            break
        end
    end
    while length(xs) < N
        x = xs[end] + rand(rng, -vmax:vmax)
        y = ys[end] + rand(rng, -vmax:vmax)
        if in_maze(x, y)
            push!(xs, x)
            push!(ys, y)
        end
    end
    (x = xs, y = ys)
end
