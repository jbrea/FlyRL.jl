struct Preprocessor{I, T, S}
    input::I
    target::T
    shock::S
    target_offset::Int
end
"""
$SIGNATURES

Define a preprocessor with input encoder `input`, target encoder `target` and
shock encoder `shock` (= `ColumnPicker(:shock)` by default).
"""
function Preprocessor(; input, target, shock = ColumnPicker(:shock), target_offset = 1)
    Preprocessor(input, target, shock, target_offset)
end
isdynamic(p::Preprocessor) = isdynamic(p.input)
markov(p::Preprocessor) = markov(p.input)
"""
$SIGNATURES

Preprocess data with preprocessor `p` (see also [`Preprocessor`](@ref)).
"""
function preprocess(p::Preprocessor, data)
    if isdynamic(p.input)
        input, idxs = encode(p.input, data)
        target = first(encode(p.target, data))
        shock = first(encode(p.shock, data))
        target = target[idxs[1+p.target_offset:end]]
        shock = [mean(shock[idxs[i]+1:idxs[i+1]]) for i in 1:length(idxs)-1]
    else
        input = first(encode(p.input, data))
        target = first(encode(p.target, data))[1+p.target_offset:end]
        shock = float.(first(encode(p.shock, data)))
    end
    idxs = [!any(ismissing.(input[i])) &&
            !any(ismissing.(target[i])) &&
            !any(ismissing.(shock[i]))
            for i in eachindex(target)] |> findall
    (input = collect(skipmissing(input[idxs])),
     target = collect(skipmissing(target[idxs])),
     shock = shock[idxs])
end
