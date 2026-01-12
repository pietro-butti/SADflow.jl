module SADflow

    include("RealNVP.jl")
        using .RealNVP
        export biject, AffineEOCoupling, log_N01, density_loss, dsum, pass_inverse

    include("FourierFeatures.jl")
        using .FourierFeatures
        export LatConv2D, FourierFeature, FourierLatConv
    
    include("Losses.jl")
        using .Losses
        export neighbor_sum, staple_sum, Actionλϕ⁴, Forceλϕ⁴, fHf
        export batched_jvp, trace_J, trace_J2
        export Stein, score_mismatch
        export phi4loss_generate, phi4loss_source, phi4loss_OTnorm, phi4loss_KL2, phi4loss_KL2stein
end
