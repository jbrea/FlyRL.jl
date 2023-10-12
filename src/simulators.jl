"""
    Environment(; preprocessor,
                  pattern = "GBB",
                  shock = shock_function(pattern),
                  state = rand_state(preprocessor, pattern, shock))

Define an environment that can be used for simulations.
"""
Base.@kwdef struct Environment{P,X,S,Y}
    preprocessor::P
    pattern::X = "GBB"
    shock::S = shock_function(pattern)
    state::Y = rand_state(preprocessor, pattern, shock)
end
function shock_function(pattern)
    pattern[1] == 'G' && return in_left
    pattern[2] == 'G' && return in_middle
    pattern[3] == 'G' && return in_right
    error("`shock_function` not defined for pattern $pattern.")
end
shock(e::Environment) = e.shock(e.state)
pattern(e::Environment) = e.pattern
function rand_state!(e::Environment; rng = Random.default_rng())
    e.state .= rand_state(e.preprocessor, e.pattern, e.shock; rng)
end
state(e::Environment) = e.state
"""
$SIGNATURES
"""
function random_track(; N = 2000,
                        pattern = "GBB",
                        shock = shock_function(pattern),
                        rng = Random.default_rng())
    x, y = random_maze_position(; N, rng)
    data = DataFrame(x = x, y = y, pattern = fill(pattern, N), t = cumsum(fill(DEFAULT_Δt, N)))
    data.shock = shock.(data.x, data.y)
    data
end
function rand_state(preprocessor, pattern, shock = shock_function(pattern);
                    rng = Random.default_rng())
    data = random_track(; pattern, shock, rng)
    first(first(preprocess(preprocessor, data)))
end
function step!(e::Environment, a)
    step!(e.preprocessor.input, e.preprocessor.target, e.state, a),
    shock(e)
end
function step!(state::AbstractVector, K::Int)
    K == 1 && return state
    for k in keys(state)
        suffix(k) == 1 || continue
        kbase = first(split_suffix(k))
        for j in K-1:-1:2
            next = add_suffix(kbase, j)
            current = add_suffix(kbase, j-1)
            setproperty!(state, next, getproperty(state, current))
        end
        setproperty!(state, k, getproperty(state, Symbol(kbase)))
    end
    state
end
# TODO: performance optimization
function step!(senc::VectorEncoder, tenc::LevelEncoder, state, action::Int)
    step!(state, markov(senc))
    for (i, k) in pairs(levels(tenc)[1])
        setproperty!(state, Symbol(k), action == i)
    end
    state
end
function step!(::VectorEncoder, tenc::DeltaPositionIndexEncoder, state, action::Int)
    dx = tenc.dx[action]
    dy = tenc.dy[action]
    state.oxˌ1 = state.ox
    state.oyˌ1 = state.oy
    state.speedˌ1 = state.speed
    state.angleˌ1 = state.angle
    state.Δx = dx
    state.Δy = dy
    state.x += dx
    state.y += dy
    encode!(OrientationEncoder(), state, state)
    encode!(AngleEncoder(), state, state)
    encode!(SpeedEncoder(), state, state)
    state
end
