module Metrics
    using Lux, Pipe, StatsBase
    using ..Utils

    dsum(x; dims) = dropdims(sum(x; dims); dims)


    Obs(ϕ::AbstractArray{T,N}; x₀=1) where {T,N} = @pipe dsum(ϕ,dims=2) |> selectdim(_,1,x₀)

    KL2(model,ps,st,z::AbstractArray{T,N},λ::T,κ::T; ns=1) where {T,N} = begin
        func = StatefulLuxLayer(model,ps,st)
        f = func(z)

        trJ² = trJJ(func,z; ns=ns)[:]
        f_Hf = fHf(z,f,λ,κ)[:]
        f∇O = sum(selectdim(f,1,1),dims=1)[:]
        
        return mean(trJ² .+ f_Hf .- 2 .* f∇O), st, (; trJJ=trJ², fHf=f_Hf, fDO=f∇O)
    end  


    export KL2, Obs
end