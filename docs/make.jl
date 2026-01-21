using Documenter, NonparametricVecchia

makedocs(
  modules = [NonparametricVecchia],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(ansicolor = true,
                           prettyurls = get(ENV, "CI", nothing) == "true",
                           collapselevel = 1),
  sitename = "NonparametricVecchia.jl",
  pages = ["Home" => "index.md",
           "Tutorials" => "vecchia_model.md"],
)

deploydocs(
  repo = "github.com/exanauts/NonparametricVecchia.jl.git",
  push_preview = true,
  devbranch = "main",
)
