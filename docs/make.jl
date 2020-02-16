using Documenter, TreatmentEffects

makedocs(modules = [TreatmentEffects], sitename = "TreatmentEffects.jl")

deploydocs(
    repo = "github.com/kmmate/TreatmentEffects.jl.git",
)