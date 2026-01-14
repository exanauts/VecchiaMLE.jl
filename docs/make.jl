using Documenter, VecchiaMLE

makedocs(
  modules = [VecchiaMLE],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(ansicolor = true,
                           prettyurls = get(ENV, "CI", nothing) == "true",
                           collapselevel = 1),
  sitename = "VecchiaMLE.jl",
  pages = ["Home" => "index.md",
    "API" => "api.md",
    "Vecchia model" => "vecchia_model.md",
    "Tutorials" => "how_to_run.md"]
)

deploydocs(
  repo = "github.com/exanauts/VecchiaMLE.jl.git",
  push_preview = true,
  devbranch = "master",
)
