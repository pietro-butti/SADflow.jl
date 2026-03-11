module SADflow
    include("Architectures/Architectures.jl")
        using .Architectures
        export biject, AffineEOCoupling, log_N01, density_loss, dsum, pass_inverse
        export LatConv2D, FourierFeature, FourierLatConv
        export div_FourierFeature, div_FourierLatConv
        export SpectralConv, FourierNeuralLayer, FNO

    include("Utils.jl")
        using .Utils
        export show_keys
        export neighbor_sum, staple_sum, Actionλϕ⁴, Forceλϕ⁴, fHf
        export vjv, vjjv, trJ, trJJ

    include("Metrics.jl")
        using .Metrics
        export KL2, Obs
end
