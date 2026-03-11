module Utils
    using Lux, NNlib, MLUtils, ForwardDiff
    using StatsBase, LinearAlgebra, Random, Pipe

    function show_keys(nt::NamedTuple, prefix="", is_last=true)
        ks = collect(keys(nt))
        for (i, key) in enumerate(ks)
            connector = i == length(ks) ? "└─ " : "├─ "
            println(prefix, connector, key)
            
            value = getfield(nt, key)
            if value isa NamedTuple
                extension = i == length(ks) ? "   " : "│  "
                show_keys(value, prefix * extension, i == length(ks))
            end
        end
    end

    ## ======================= λϕ⁴-specific func =========================
        neighbor_sum(ϕ::AbstractArray{T,4}) where T = 
            circshift(ϕ,(-1,0,0,0)) .+ 
            circshift(ϕ,(0,-1,0,0)) 
        
        neighbor_sum(ϕ::AbstractArray{T,6}) where T = 
            circshift(ϕ,(-1,0,0,0,0,0)) .+ 
            circshift(ϕ,(0,-1,0,0,0,0)) .+ 
            circshift(ϕ,(0,0,-1,0,0,0)) .+ 
            circshift(ϕ,(0,0,0,-1,0,0))

        staple_sum(ϕ::AbstractArray{T,4}) where T = 
            circshift(ϕ,(-1,0,0,0)) .+ circshift(ϕ,(1,0,0,0)) .+ 
            circshift(ϕ,(0,-1,0,0)) .+ circshift(ϕ,(0,1,0,0))    
        
        staple_sum(ϕ::AbstractArray{T,6}) where T = 
            circshift(ϕ,(-1,0,0,0,0,0)) .+ circshift(ϕ,(1,0,0,0,0,0)) .+ 
            circshift(ϕ,(0,-1,0,0,0,0)) .+ circshift(ϕ,(0,1,0,0,0,0)) .+ 
            circshift(ϕ,(0,0,-1,0,0,0)) .+ circshift(ϕ,(0,0,1,0,0,0)) .+ 
            circshift(ϕ,(0,0,0,-1,0,0)) .+ circshift(ϕ,(0,0,0,1,0,0))   


        Actionλϕ⁴(ϕ::AbstractArray{T,N},λ::T,κ::T) where {T,N} = (
            -2κ .* ϕ .* neighbor_sum(ϕ) .+ ϕ.^2 .+ λ .* (ϕ.^2 .- T(1)).^2
        ) |> x->sum(x; dims=1:N-1)

        Forceλϕ⁴(ϕ::AbstractArray{T,N},λ::T,κ::T) where {T,N} =
            -2κ .* staple_sum(ϕ) .+ 2ϕ .+ 4λ .* (ϕ.^2 .- T(1)) .* ϕ

        fHf(ϕ::AbstractArray{T,N},f::AbstractArray{T,N},λ::T,κ::T) where {T,N} = (
            -2κ .* f .* staple_sum(f) .+ (T(2) - 4λ .+ 12λ .* ϕ .^ 2) .* f .^2
        ) |> x->sum(x; dims=1:N-1)

    ## ===================================================================

    ## ======================= Hutchinson's trace =========================
        vjv = (rng, func, z)-> begin
            T = eltype(z)
            η = T.(rand(rng,[-1,1],size(z)...))  # Match type of z
            Jη = jacobian_vector_product(func, AutoForwardDiff(), z, η)
            @assert ndims(η)==ndims(Jη)
            eachslice(η .* Jη,dims=ndims(η)) .|> sum
        end

        trJ(rng, func, z; ns=1) = 
            [vjv(rng,func,z) for _ in 1:ns] |> stack
        trJ(func, z; ns=1) = 
            trJ(Random.default_rng(1), func,z; ns=ns)

        vjjv = (rng, func, z)-> begin
            T = eltype(z)
            η = rand(rng,[-1,1],size(z)...) .|> T  # Match type of z
            Jη  = jacobian_vector_product(func, AutoForwardDiff(), z, η)
            JJη = jacobian_vector_product(func, AutoForwardDiff(), z, Jη)
            @assert ndims(η)==ndims(JJη)
            eachslice(η .* JJη,dims=ndims(η)) .|> sum
        end

        trJJ(rng, func, z; ns=1) = 
            @pipe [vjjv(rng,func,z) for _ in 1:ns] |> stack |> mean(_,dims=2)
        trJJ(func, z; ns=1) = 
            trJJ(Random.default_rng(1), func,z; ns=ns)
    ## ====================================================================


    export show_keys
    export neighbor_sum, staple_sum, Actionλϕ⁴, Forceλϕ⁴, fHf
    export vjv, vjjv, trJ, trJJ
end