using ExaModels, MadNLP, JuMP, PowerModels

include("exa_from_file.jl")

# pglib_opf_case118_ieee from .nl file
case118nl = exa_from_file("pglib_opf_case118_ieee.nl")
madnlp(case118nl)

# pglib_opf_case118_ieee from .m file
case118m = exa_from_file("pglib_opf_case118_ieee.m")
madnlp(case118m)

# Solve w/ GPU
using CUDA, MadNLPGPU

# pglib_opf_case118_ieee from .m file w/ GPU
case118gpu = exa_from_file("pglib_opf_case118_ieee.m", backend = CUDABackend())
madnlp(case118gpu)