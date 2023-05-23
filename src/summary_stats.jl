# API
# summarize(stats, track)

struct RelativeTimeInState{E}
    encoder::E
    key::String
    exclude::Vector{String}
end
"""
$SIGNATURES

Compute the relative duration spent in `state` in the encoding defined by the `encoder`.
States can be excluded with `exclude = ["state1", "state2"]`.

## Example
```
summarize(RelativeTimeInState(SemanticEncoder7(), "left turn", exclude = ["center"]), track)
```
"""
RelativeTimeInState(encoder, state; exclude = String[]) = RelativeTimeInState(encoder, state, exclude)
labels(s::RelativeTimeInState) = (Symbol("relative_time_in_", s.key),)
chance_level(e::RelativeTimeInState) = 1/(length(levels(e.encoder)[1]) - length(e.exclude))
function summarize(s::RelativeTimeInState, track)
    data, Δt = encode(DurationPerStateEncoder(s.encoder), track)
    idxs = findall(==(s.key), data)
    idxs_exclude = findall(∈(s.exclude), data)
    NamedTuple{labels(s)}(sum(Δt[idxs])/(sum(Δt) - sum(Δt[idxs_exclude])))
end
"""
$SIGNATURES

Relative time in shock arm.
"""
RelativeTimeInShockArm() = RelativeTimeInState(ShockArmEncoder(), "shock", exclude = ["center"])

struct RelativeVisitsToState{E}
    encoder::E
    key::String
    exclude::Vector{String}
end
"""
$SIGNATURES

Compute the relative number of visits to `state` in the encoding defined by the `encoder`.
States can be excluded with `exclude = ["state1", "state2"]`.

## Example
```
summarize(RelativeVisitsToState(SemanticEncoder7(), "left turn", exclude = ["center"]), track)
```
"""
RelativeVisitsToState(e, key; exclude = String[]) = RelativeVisitsToState(e, key, exclude)
labels(s::RelativeVisitsToState) = (Symbol("relative_visits_to_", s.key),)
chance_level(e::RelativeVisitsToState) = 1/(length(levels(e.encoder)[1]) - length(e.exclude))
function summarize(s::RelativeVisitsToState, track)
    data, = encode(DynamicCompressEncoder(labels(s.encoder)[1], s.encoder), track)
    idxs = findall(==(s.key), data)
    idxs_exclude = findall(∈(s.exclude), data)
    NamedTuple{labels(s)}((length(idxs)/(length(data) - length(idxs_exclude)),))
end
"""
$SIGNATURES

Relative visits to shock arm.
"""
RelativeVisitsToShockArm() = RelativeVisitsToState(ShockArmEncoder(), "shock", ["center"])

struct ChangeOf{S}
    stats::S
    midpoint::Float64
end
"""
$SIGNATURES

Compute the change in statistics `stat` before and after the `midpoint` (= 0.5 by default).
## Example
```
summarize(ChangeOf(RelativeTimeInShockArm()), track)
```
"""
ChangeOf(stat, midpoint = 0.5) = ChangeOf{typeof(stat)}(stat, midpoint)
chance_level(::ChangeOf) = 0.
labels(s::ChangeOf) = (Symbol("changeof_", labels(s.stats)[1]),)
function summarize(s::ChangeOf, track)
    T = floor(Int, nrow(track)*s.midpoint)
    s1, = summarize(s.stats, track[1:T, :])
    s2, = summarize(s.stats, track[T+1:end, :])
    NamedTuple{labels(s)}((s2 - s1,))
end
