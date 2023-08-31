####
#### Layer
####
struct DenseLayer{F,T}
    func::F
    output::Matrix{T}
    w::Matrix{T}
    b::Vector{T}
end
Base.zero(l::DenseLayer) = DenseLayer(l.func, zero(l.output), zero(l.w), zero(l.b))
function zero!(l::DenseLayer)
    zero!(l.output)
    zero!(l.w)
    zero!(l.b)
end
"""
$SIGNATURES

Makes a dense layer.
"""
function DenseLayer(Din, Dout, f; T = Float64, w = glorot_normal(Din, Dout, T), b = zeros(T, Dout), batchsize = 1)
    DenseLayer(f, zeros(T, Dout, batchsize), w, b)
end
function (d::DenseLayer{F,T})(x) where {F,T}
    w = d.w
    b = d.b
    f = d.func
#     out = view(d.output, axes(w, 1), axes(x, 2))
    out = d.output
    out .= w * x # Enzyme doesn't like mul!(out, w, x) with views for some reason
    @. out = f(out)
    out
end
params(l::DenseLayer) = ComponentArray(w = l.w, b = l.b)

####
#### Init
####
relu(x) = ifelse(x < 0, zero(x), x)
glorot_normal(in, out, T = Float64) = glorot_normal(Random.GLOBAL_RNG, in, out, T)
glorot_uniform(in, out, T = Float64) = glorot_uniform(Random.GLOBAL_RNG, in, out, T)
glorot_normal(rng::AbstractRNG, in, out, T = Float64) = randn(rng, T, out, in) * sqrt(T(2)/(in + out))
glorot_uniform(rng::AbstractRNG, in, out, T = Float64) = (rand(rng, T, out, in) .- T(0.5)) * T(2) * sqrt(T(6)/(in + out))

####
#### MLP
####

struct MLP{L,P}
    layers::L
end
"""
$SIGNATURES

Makes a multi-layer perceptron.
## Example
```
mlp = MLP(DenseLayer(10, 20, relu), DenseLayer(20, 1, identity))
mlp(rand(10))
```
"""
MLP(layers...) = MLP{typeof(layers),paramlabels(layers)}(layers)
Base.zero(m::MLP) = MLP(zero.(m.layers)...)
function zero!(m::MLP)
    for l in m.layers
        zero!(l)
    end
    m
end
paramlabels(::DenseLayer) = ("w", "b")
function paramlabels(layers) # should be adapted for other layer types
    tuple((Symbol.(paramlabels(l) .* "$i") for (i, l) in pairs(layers))...)
end
function params(mlp::MLP{L,P}) where {L, P}
    ks = tuple(Iterators.flatten(P)...)
    vs = tuple(Iterators.flatten((copy(l.w), copy(l.b)) for l in mlp.layers)...)
    ComponentArray(NamedTuple{ks}(vs))
end
initialize!(mlp::MLP{L,P}, params) where {L,P} = initialize!(mlp.layers, P, params)
initialize!(::Tuple{}, ::Any, ::Any) = nothing
@inline function initialize!(layers, P, params)
    initialize!(first(layers), first(P), params)
    initialize!(Base.tail(layers), Base.tail(P), params)
end
@inline function initialize!(d::DenseLayer, P, params)
    d.w .= getproperty(params, P[1])
    d.b .= getproperty(params, P[2])
end
function initialize!(d::DenseLayer, params)
    d.w .= params.w
    d.b .= params.b
end
(mlp::MLP)(x) = forward!(first(mlp.layers), Base.tail(mlp.layers), x)
forward!(layer, layers, x) = forward!(first(layers), Base.tail(layers), layer(x))
forward!(layer, ::Tuple{}, x) = layer(x)
