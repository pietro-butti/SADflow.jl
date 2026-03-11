module Architectures
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
end