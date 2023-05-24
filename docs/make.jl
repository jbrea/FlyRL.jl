# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, FlyRL

makedocs(
    modules = [FlyRL],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Johanni Brea",
    sitename = "FlyRL.jl",
    pages = Any["Introduction" => "index.md",
                "Examples" => [
                               "Abstract Model" => "abstract_model.md",
                               "Detailed Model" => "detailed_model.md"
                              ],
                "API" => [
                          "Loading Files" => "io.md",
                          "Encoders" => "encoders.md",
                          "Models" => "models.md",
                          "Fitting" => "fitting.md",
                          "Simulations" => "simulations.md",
                          "Summary Statistics" => "summary_stats.md"
                         ]
               ]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

# Some setup is needed for documentation deployment, see “Hosting Documentation” and
# deploydocs() in the Documenter manual for more information.
deploydocs(
    repo = "github.com/jbrea/FlyRL.jl.git",
    push_preview = true
)
