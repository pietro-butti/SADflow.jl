module SADflow

    include("RealNVP.jl")
        using .RealNVP
        export biject, AffineEOCoupling, log_N01, density_loss, dsum, pass_inverse

end
