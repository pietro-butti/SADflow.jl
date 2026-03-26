module Utils
    using Lux, NNlib, MLUtils, ForwardDiff
    using FFTW
    using StatsBase, LinearAlgebra, Random, Pipe
    using FormalSeries, ADerrors

    import Base: length, iterate
    length(x::uwreal) = 1
    iterate(x::uwreal) = (x, nothing)
    iterate(x::uwreal,::Nothing) = nothing

    dsum(x; dims) = dropdims(sum(x; dims); dims)


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
    ## --- Implicit functions for 2D -------------------------------------
        _action(λ,κ) = ϕ ->
            ϕ.^2 .+ λ .* (ϕ.^2 .- 1).^2 .+
            -2κ .* ϕ .* (
                circshift(ϕ,(-1,0)) .+ circshift(ϕ,(0,-1))
            ) |> sum

        _grad_S(λ,κ) = ϕ -> 
            2ϕ .+ 4λ .* (ϕ.^2 .- 1) .* ϕ .+ 
            -2κ .* (
                circshift(ϕ,(-1,0)) .+ circshift(ϕ,(1,0)) .+
                circshift(ϕ,(0,-1)) .+ circshift(ϕ,(0,1)) 
            )

        _Hvp(λ,κ) = (ϕ,v) -> 
            (2 .- 4λ .+ 12λ .* ϕ.^2) .* v .+ 
            -2κ .* (
                circshift(v,(-1,0)) .+ circshift(v,(1,0)) .+
                circshift(v,(0,-1)) .+ circshift(v,(0,1)) 
            )
        
        obs(z; x₀=1) = dsum(z,dims=2)[x₀]         
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
            trJ(Random.default_rng(), func,z; ns=ns)

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
            trJJ(Random.default_rng(), func,z; ns=ns)
    ## ====================================================================

    ## ========================= Langevin methods ================================
        LangevinStep(Zₜ, ∇S, Δt, ηₜ) = Zₜ .- Δt .* ∇S .+ sqrt(2Δt)*ηₜ
        AdjointLangevinStep(Aₛ, Hₛv, Δt, d₀)   = Aₛ .+ Δt .* (-Hₛv .+ d₀)

        function FeynmanKac(z₀,η,δt; grad_S=nothing, Hvp=nothing, δ₀=nothing)
            itr = eachslice(η,dims=ndims(η))
            (Z_T,traj) = foldl(itr; init=(z₀,())) do (Zₜ,traj), ηₜ
                Zₜ = LangevinStep(Zₜ,grad_S(Zₜ),δt,ηₜ)
                (Zₜ,(traj...,Zₜ))
            end

            A₀ = zeros(eltype(Z_T),size(Z_T)...)
            int = foldl(reverse(traj); init=A₀) do Aₛ, Zₜ
                HₛA = Hvp(Zₜ,Aₛ)
                Aₛ = AdjointLangevinStep(Aₛ,HₛA,δt,δ₀)
            end

            return int
        end
    ## ==========================================================================


    ## ========================= AD reweigthing ===========================
        # Compute the free scalar propagator on a 1D momentum grid and
        # broadcast it into the full lattice volume.
        #
        # FFTW conventions used throughout:
        #   fft(f)[k]  = Σ_x exp(-2πi/N · k·x) f(x)   (no 1/V prefactor)
        #   ifft(f)[x] = (1/V) Σ_k exp(+2πi/N · k·x) f(k)
        function free_propagator(vol, κ, λ; ft=ifft)
            m̂² = (1-2λ)/κ - 2*length(vol)
            p   = 2π / first(vol) .* ((1:first(vol)) .- 1)
            fp  = @. 1/2 / ((2 * sin(p/2))^2 + m̂²)

            f̃       = zeros(vol...)
            f̃[:,1] .= fp
            return f̃ |> ft .|> real
        end        

        function sumnorm(x::AbstractArray{T,N};dims=2) where {T,N}
            c = dsum(x,dims=dims)
            c = map(eachcol(c)) do x; x ./ first(x) end |> stack
        end

        Δf(f0) = fout -> begin
            c = sumnorm(fout)
            f = sumnorm(f0)
            abs.(c .- f) ./ f
        end

        deltalike(z; xstar=1) = begin
            δ = zeros(eltype(z),size(z)...)
            selectdim(δ,1,xstar) .= one(eltype(z))
            return δ
        end
        

        function compute_weight(z::AbstractArray{T,N},f,action,obs,trJf) where {T,N}
            # Compute improved confs
            ε = Series((T(0.),T(1.)))
            ϕ̃ = z .+ ε .* f

            cfg = eachslice(z,dims=N)
            itr = eachslice(ϕ̃,dims=N)

            S̃ = action.(itr) .- ε .* obs.(itr)
            ΔS̃ = S̃ .- action.(cfg)    # ε (f⋅∇S - O₀)

            return exp.(-ΔS̃ .- ε.*trJf)
        end

        function reweight_corr(w̃,ϕ̃,uwargs...)
            # 1/(a+b) = 1/a * 1/(1+b/a) = 1/a * (1-b/a) = 1/a - b/a^2 = 
            w₀ = uwreal(Float64.(getindex.(w̃,1)),uwargs...)
            w₁ = uwreal(Float64.(getindex.(w̃,2)),uwargs...)

            Ew = reshape(w̃,1,:) .* dsum(ϕ̃,dims=2)[:,1,:]

            # Extract both ε-orders of numerator
            num₀ = map(eachrow(Ew)) do r
                uwreal(Float64.(getindex.(r, 1)[:]), uwargs...)    # ⟨Φ₀⟩ = ⟨w₀ φ⟩
            end
            num₁ = map(eachrow(Ew)) do r
                uwreal(Float64.(getindex.(r, 2)[:]), uwargs...)    # ⟨Φ₁⟩ = ⟨w₁φ + w₀f(φ)⟩
            end

            # C(y₀) = ⟨Φ₁⟩/⟨w₀⟩ - ⟨Φ₀⟩⟨w₁⟩/⟨w₀⟩²
            corri  = @. num₁ / w₀ - num₀ * w₁ / w₀^2

            return corri
        end
    ## =====================================================================


    export show_keys
    export neighbor_sum, staple_sum, Actionλϕ⁴, Forceλϕ⁴, fHf
    export vjv, vjjv, trJ, trJJ
    export free_propagator, sumnorm, reweight_corr, compute_weight
    export _action, _grad_S, _Hvp, deltalike, obs, Δf, LangevinStep, AdjointLangevinStep, FeynmanKac
    
end