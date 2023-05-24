Encoders can be used to transform the raw data in different ways.
This can be done in two ways, illustrated here with the [`ShockArmEncoder`](@ref):
1. Create a new data frame with the transformed data
```
using DataFrames
import FlyRL: encode, ShockArmEncoder, random_track
track = random_track(N = 100)
encode(ShockArmEncoder(), track) |> DataFrame
```
2. Append the transformed data directly to the original data frame
```
using DataFrames
import FlyRL: encode!, ShockArmEncoder, random_track
track = random_track(N = 100)
encode!(ShockArmEncoder(), track)
```

Here is a list of built-in encoders.

```@autodocs
Modules = [FlyRL]
Pages = ["io.jl"]
```
