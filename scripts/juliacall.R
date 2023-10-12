setup.jl <- function (installJulia = False) {
    install.packages("JuliaCall")
    library(JuliaCall)
    julia_setup(installJulia = installJulia)
    julia_eval("using Pkg; Pkg.activate(\"YMazeJL\"); Pkg.add(url = \"https://github.com/jbrea/FlyRL.jl\", subdir=\"lib/YMaze\")")
}

load.jl <- function () {
    library(JuliaCall)
    julia_eval("using Pkg; Pkg.activate(\"YMazeJL\"); using YMaze")
}

read <- function (track, args = "") {
    julia_eval(paste0("YMaze.read(\"", track, "\" ;", args, ")"))
}

read_directory <- function(dir, args = "") {
    julia_eval(paste0("YMaze.read_directory(\"", dir, "\"; ", args, ")"))
}
