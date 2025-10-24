"""
    exa_from_file(
        filename::String
        backend = nothing
    )

Returns an ExaModel read from `filename` in formats `.cbf`, `.lp`, `.mof`, `.mps`, `.nl`, `.rew`, or `.sdpa`.

Reading `.m`, `.raw`, or `.json` files requires `PowerModels.jl`.

Available backends provided by the JuliaGPU ecosystem:\n
- `CPU()`: multi-threaded CPU Execution
- `CUDABackend()` : NVIDIA GPUs (`CUDA.jl`)
- `ROCBackend()` : AMD GPUs (`AMDGPU.jl`)
- `OneAPIBackend()` : Intel GPUs (`oneAPI.jl`)
- `MetalBackend()` : Apple GPUs (`Metal.jl`)
- `OpenCLBackend()` : generic OpenCL devices (`OpenCL.jl`)

## Example

```jldoctest
julia> m = exa_from_file("pglib_opf_case118_ieee.m", backend = CUDABackend())
An ExaModel{Float64, CuArray{Float64, 1, CUDA.DeviceMemory}, ...}
[...]

julia> madnlp(m)
"Execution stats: Optimal Solution Found (tol = 1.0e-08)."
```
"""
function exa_from_file(
    filename::Core.String;
    backend = nothing
    )
    if lowercase(last(split(filename, '.'))) in ["m", "raw", "json"]
        pm = PowerModels.instantiate_model(
            PowerModels.parse_file(filename), 
            PowerModels.ACPPowerModel, 
            PowerModels.build_opf
        )
        filename = Base.Filesystem.joinpath(mktempdir(), "pm.nl")
        JuMP.write_to_file(pm.model, filename)
    end
    return ExaModels.ExaModel(JuMP.read_from_file(filename; use_nlp_block = false); backend = backend)
end