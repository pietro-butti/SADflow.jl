module Metrics
    using Lux, Pipe, StatsBase, Random
    using ..Utils

    dsum(x; dims) = dropdims(sum(x; dims); dims)


    Obs(ϕ::AbstractArray{T,N}; x₀=1) where {T,N} = @pipe dsum(ϕ,dims=Tuple(2:N-1)) |> selectdim(_,1,x₀)

    KL2(
        model,ps,st,z::AbstractArray{T,N},λ::T,κ::T;
        ns=1, 
        rng=Random.default_rng(),
        div_f2=nothing
    ) where {T,N} = begin
        func = StatefulLuxLayer(model,ps,st)
        f = func(z)

        trJ² = isnothing(div_f2) ? trJJ(rng, func, z; ns=ns)[:] : div_f2(model,ps,st,z)
        f_Hf = fHf(z,f,λ,κ)[:]
        f∇O = sum(selectdim(f,1,1),dims=1)[:]
        
        return mean(trJ² .+ f_Hf .- 2 .* f∇O), st, (; trJ², f_Hf, f∇O)
    end  


    KL2sq(model,ps,st,z::AbstractArray{T,N},λ::T,κ::T; ns=1, rng=Random.default_rng(), Oav=nothing) where {T,N} = begin
        func = StatefulLuxLayer(model,ps,st)
        f = func(z)

        div_f = trJ(rng, func,z; ns=ns)[:]
        f∇S   = sum(Forceλϕ⁴(z,λ,κ) .* f,dims=1:N-1)[:]

        O₀ = sum(selectdim(z, 1, 1); dims=1:N-2)[:]
        ΔO₀ = O₀ .- (isnothing(Oav) ? mean(O₀) : Oav)

        res = (div_f .- f∇S) .+ ΔO₀
        
        return mean(res .^ 2), st, (; trJ=div_f, f∇S=f∇S)
    end  


    export KL2, KL2sq, Obs
end