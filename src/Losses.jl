module Losses
    using Lux, NNlib, MLUtils
    using StatsBase

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
            -2κ .* staple_sum(ϕ) .+ 2ϕ + 4λ .* (ϕ.^2 .- T(1)) .* ϕ

        fHf(ϕ::AbstractArray{T,N},f::AbstractArray{T,N},λ::T,κ::T) where {T,N} = (
            -4κ .* f .* neighbor_sum(f) .+ (T(2) - 4λ .+ 12λ .* ϕ .^ 2) .* f .^2
        ) |> x->sum(x; dims=1:N-1)
    ## ===================================================================

    ## ======================= Hutchinson's trace =========================
        batched_jvp = (func,z) -> begin
            η = rand([-1,1],size(z)...)
            v = jacobian_vector_product(func, AutoForwardDiff(), z, η)
            sum(η .* v, dims=(1,2))[:]
        end

        trace_J = (func,z,nsources) -> begin
            [batched_jvp(func,z) for _ in 1:nsources] |> stack
        end
        
        trace_J2 = (func,z) -> begin
            η = rand([-1,1],size(z)...)
            v = jacobian_vector_product(func, AutoForwardDiff(), z, η)
            w = jacobian_vector_product(func, AutoForwardDiff(), z, v)
            sum(η .* w, dims=(1,2))[:]
        end
    ## ====================================================================

    function Stein(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int) where {T,N}
        func = StatefulLuxLayer(model,ps,st)
        f = func(ϕ)
        
        # Divergence: ∇⋅f = tr J_f 
        divf = mean(trace_J(func,ϕ,nsources),dims=2)[:]
    
        # Score term f⋅∇log(r) = - f⋅∇S 
        f∇S = mean(Forceλϕ⁴(ϕ,λ,κ) .* f, dims=collect(1:N-1))[:]

        return f, divf .- f∇S
    end

    function score_mismatch(f::AbstractArray{T,N}, x₀::Int) where {T,N}
        return .- sum.(eachslice(selectdim(f,1,x₀),dims=N-1)) 
    end
    ##


    function phi4loss_generate(model, ps, st, z::AbstractArray{T,N}, λ::T,κ::T) where {T,N}
        zero_batch = zeros_like(z, size(z,N))
        (ϕ, logdetJ), st = model((z, zero_batch), ps, st)

        logp = -Actionλϕ⁴(ϕ,λ,κ)[:]
        logr = sum(log_N01(z); dims=collect(1:N-1))[:]
        logq = logr - logdetJ

        logw = logp - logq
        log_ess = 2*logsumexp(logw) - logsumexp(2* logw)
        ess = exp(log_ess)/length(logw)

        return mean(logq - logp), st, (; ess=ess)
    end

    function phi4loss_source(model, ps, st, z::AbstractArray{T,N}, λ::T,κ::T,J::T; x₀=1) where {T,N}
        zero_batch = zeros_like(z, size(z,N))
        (ϕ, logdetJ), st = model((z, zero_batch), ps, st)

        logq = .- Actionλϕ⁴(z,λ,κ)[:] .- logdetJ

        JO = .- J .* sum(selectdim(ϕ,1,x₀),dims=1)[:]
        logp = .- Actionλϕ⁴(ϕ,λ,κ)[:] + JO

        logw = logp - logq
        log_ess = 2*logsumexp(logw) - logsumexp(2* logw)
        ess = exp(log_ess)/length(logw)

        return mean(logq .- logp), st, (; ess=ess)
    end

    function phi4loss_OTnorm(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int, x₀::Int; O_av=nothin) where {T,N}
        f,Tr_f = Stein(model,ps,st,ϕ,λ,κ,nsources) # Stein operator

        # Control variates
        O = .- sum.(eachslice(selectdim(ϕ,1,x₀),dims=N-1))
        Op_av = isnothing(O_av) ? mean(O) : O_av
        Ō = O .- Op_av

        return norm((Tr_f .- Ō).^2,2), st, (;)
    end

    function phi4loss_KL2(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int, x₀::Int) where {T,N}
        func = StatefulLuxLayer(model,ps,st)
        f = func(ϕ)
        
        # Hessian term
        f⊥Hf = fHf(ϕ,f,λ,κ)[:] 

        # Stochastic Jacobian trace
        trJ² = mean(
            [trace_J2(func,ϕ) for _ in 1:nsources] |> stack,
            dims=2
        )[:]

        # Score mismatch
        f∇O = score_mismatch(f,x₀)

        # Variance term
        σₒ² = sum.(eachslice(selectdim(ϕ,1,x₀),dims=ndims(ϕ)-1)) |> var

        return mean(trJ²./2 .+ f⊥Hf./2 .+ f∇O) + σₒ²/2, st, (;)
    end

    function phi4loss_KL2stein(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int, x₀::Int) where {T,N}
        func = StatefulLuxLayer(model,ps,st)
        f = func(ϕ)
        
        # Hessian term
        f⊥Hf = fHf(ϕ,f,λ,κ)[:] 

        # Stochastic Jacobian trace
        trJ² = mean(
            [trace_J2(func,ϕ) for _ in 1:nsources] |> stack,
            dims=2
        )[:]

        # Score mismatch
        f∇O = score_mismatch(f,x₀)

        # Variance term
        σₒ² = sum.(eachslice(selectdim(ϕ,1,x₀),dims=ndims(ϕ)-1)) |> var

        return mean(trJ²./2 .+ f⊥Hf./2 .+ f∇O) + σₒ²/2, st, (;)
    end

    export neighbor_sum, staple_sum, Actionλϕ⁴, Forceλϕ⁴, fHf
    export batched_jvp, trace_J, trace_J2
    export Stein, score_mismatch
    export phi4loss_generate, phi4loss_source, phi4loss_OTnorm, phi4loss_KL2, phi4loss_KL2stein
end