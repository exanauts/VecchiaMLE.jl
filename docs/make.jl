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
    "Tutorials" => "examples/HowToRun.md"]

)

deploydocs(
  repo = "github.com/exanauts/VecchiaMLE.jl.git",
  push_preview = true,
  devbranch = "master",
)
