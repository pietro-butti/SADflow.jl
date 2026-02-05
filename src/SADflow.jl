module SADflow

    include("RealNVP.jl")
        using .RealNVP
        export biject, AffineEOCoupling, log_N01, density_loss, dsum, pass_inverse

    include("FourierFeatures.jl")
        using .FourierFeatures
        export LatConv2D, FourierFeature, FourierLatConv
        export div_FourierFeature, div_FourierLatConv

    include("FourierNeuralOperator.jl")
        using .FourierNeuralOperator
        export SpectralConv, FourierNeuralLayer, FNO


    include("Utils.jl")
        using .Utils
        export show_keys
        export neighbor_sum, staple_sum, Actionλϕ⁴, Forceλϕ⁴, fHf
        export batched_jvp, trace_J, trace_J2
        export Stein, score_mismatch
        export phi4loss_generate, phi4loss_source, phi4loss_OTnorm, phi4loss_KL2, phi4loss_KL2stein
end
