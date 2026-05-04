module SADflow
    include("Architectures/Architectures.jl")
        using .Architectures
        export biject, AffineEOCoupling, log_N01, density_loss, dsum, pass_inverse
        export SeriesConv
        export LatConv2D, FourierFeature, FourierLatConv
        export div_FourierFeature, div_FourierLatConv
        export SpectralConv, FourierNeuralLayer, FNO

    include("Utils/Utils.jl")
        using .Utils
        export dmean, sumvol, batched_sumvol, show_keys
        export batched_fft, batched_ifft, stack_complex, riffle_complex, build_source
        export uwcorr, meff, plottable
        export vjv, vjjv, trJ, trJJ
        export nn, _action, _grad_S, _Hvp, _fHf, propagator, source
    
    # include("Metrics.jl")
    #     using .Metrics
    #     export KL2, KL2sq, Obs
end
