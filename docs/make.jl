# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, FlyRL

makedocs(
    modules = [FlyRL],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Johanni Brea",
    sitename = "FlyRL.jl",
    pages = Any["index.md", "api.md"]
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
