module RealNVP
    using ConcreteStructs
    using Lux, NNlib, MLUtils
    using StatsBase, Random


    ## ------------------------ Helpers ------------------------
        biject(z::AbstractArray{T,N}; dim=1) where {T,N} = begin
            n = size(z, dim) ÷ 2
            idx1 = ntuple(Returns(Colon()), dim-1)
            idx2 = ntuple(Returns(Colon()), N-dim)
            z[idx1...,1:n,idx2...], z[idx1...,(n+1):end,idx2...]
        end

        dsum(x; dims) = dropdims(sum(x; dims); dims)
    ## ---------------------------------------------------------

    ## ----------------- Affine checkerboard coupling -----------------
        @concrete struct AffineEOCoupling{NN} <: AbstractLuxContainerLayer{(:net,)}
            net::NN
            dims::Tuple
            iseven::Bool
        end

        function Lux.initialstates(rng::AbstractRNG, l::AffineEOCoupling)
            parity = l.iseven ? iseven : isodd
            mask = [parity(sum(Tuple(I))) for I in CartesianIndices(l.dims)]

            return (; mode=:forward, mask , net=Lux.initialstates(rng, l.net))
        end

        apply_and_log_det(::Val{:forward}, s, t, z) = (z .* exp.(s) .+ t, s)
        apply_and_log_det(::Val{:inverse}, s, t, z) = ((z .- t) ./ exp.(s), -s)
        # apply_and_log_det(::Val{:forward}, s, t, z) = (z .* softplus.(s) .+ t, log.(softplus.(s)))
        # apply_and_log_det(::Val{:inverse}, s, t, z) = ((z .- t) ./ softplus.(s), -log.(softplus.(s)))

        function (cl::AffineEOCoupling)((ϕ, logdetJ), ps, st)
            # Split active/passive according to mask
            ϕ_active = ifelse.(st.mask, ϕ, zero(ϕ))
            ϕ_frozen = ifelse.(st.mask, zero(ϕ), ϕ)

            # Compute scale and shift
            (logs, t), st_net = cl.net(ϕ_frozen, ps.net, st.net)

            # Affine coupling layer
            ϕ_active, log_det = apply_and_log_det(Val(st.mode), logs, t, ϕ_active)

            # Mount back dofs        
            ϕ = ifelse.(st.mask, ϕ_active, ϕ)
            log_det = log_det .* st.mask

            # Sum local log_det (leave batch dim untouched)
            ddims = Tuple(1:ndims(st.mask))
            logdetJ += dsum(log_det; dims=ddims)

            return (ϕ, logdetJ), merge(st, (net=st_net,))
        end
    ## ----------------------------------------------------------------

    pass_inverse(model,ps,st) = (z,logdetJ) -> begin
        for (cl,p,s) in zip(model.layers, ps, st) |> Iterators.reverse        
            new_s = merge(s, (;mode=:inverse))
            (z,logdetJ),_ = cl((z,logdetJ), p, new_s)
        end
        (z,logdetJ)
    end

    ## --------------------------------- loss function --------------------------------- 
        log_N01(x::AbstractArray{T}) where {T} = -T(0.5 * log(2π)) .- T(0.5) .* abs2.(x)

        function density_loss(model, ps, st, x::AbstractArray{T,N}) where {T,N}
            # Infer dimensionality of input and batch dimension
            batch_dim = size(x,N)
            ddims = Tuple(1:(N-1))

            # Forward pass
            zero_batch = zeros_like(x, batch_dim)
            (z, logdetJ), st = model((x, zero_batch),ps,st)

            # Add prior
            logdetJ += dsum(log_N01(z); dims=ddims)

            return -mean(logdetJ), st, (;)
        end
    ## ----------------------------------------------------------------------------------


    export biject, AffineEOCoupling, log_N01, density_loss, dsum, pass_inverse
end