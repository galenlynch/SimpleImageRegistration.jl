using SimpleImageRegistration
using Documenter

DocMeta.setdocmeta!(SimpleImageRegistration, :DocTestSetup, :(using SimpleImageRegistration); recursive=true)

makedocs(;
    modules=[SimpleImageRegistration],
    authors="Galen Lynch <galen@galenlynch.com>",
    sitename="SimpleImageRegistration.jl",
    format=Documenter.HTML(;
        canonical="https://galenlynch.github.io/SimpleImageRegistration.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/galenlynch/SimpleImageRegistration.jl",
    devbranch="main",
)
