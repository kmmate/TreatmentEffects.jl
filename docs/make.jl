using Documenter, TreatmentEffect

makedocs(modules = [TreatmentEffect], sitename = "TreatmentEffect.jl")

deploydocs(
    repo = "github.com/kmmate/TreatmentEffect.jl.git",
)