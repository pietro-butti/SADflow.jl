module FourierNeuralOperator

    using Lux, NNlib
    using LuxCore, Random
    using FFTW

    ## ======================= Spectral Convolution =======================
        struct SpectralConv{F1} <: LuxCore.AbstractLuxLayer
            size::Tuple
            n_features::Int
            modes_ratio::AbstractFloat
            init_weight::F1
        end
        SpectralConv(vol::Tuple,nch::Int,r; init_weight=randnC32) = 
            SpectralConv{typeof(init_weight)}(vol,nch,r,init_weight)
        SpectralConv(v,c) = SpectralConv(v,c,1.)

        LuxCore.initialparameters(rng::AbstractRNG, n::SpectralConv) = begin
            idx = (Int.(n.size .* n.modes_ratio)...,n.n_features,n.n_features)
            return (;weight = n.init_weight(rng, idx...))
        end

        function (n::SpectralConv)(x̃::AbstractArray{T,N}, ps, st) where {T,N}
            D = N-2
            Ks = size(ps.weight)[1:D]

            u = x̃[(1:k for k in Ks)...,:,:]
            Wu = eachslice(ps.weight,dims=(1,2)) .* eachslice(u,dims=(1,2,4)) |> stack
            up = permutedims(Wu, (collect((1:D).+1)...,1,N)) #    

            return up, st
        end
    ## ====================================================================


    FourierNeuralLayer(vol,C,r; activation=σ) = Chain((
        fourier = Parallel(+;
            spectral = Chain((
                fft   = WrappedFunction(fft),
                sconv = SpectralConv(vol,C,r),
                ifft  = WrappedFunction(ifft) 
            )), 
            skip = Conv((1,1),C=>C,use_bias=false)
        ),
        activation = WrappedFunction(activation)
    ))

    FNO(vol,C,r,n_blocks; activation=σ) = Chain((
        lifting = Conv((1,1),1=>C,use_bias=false),
        NamedTuple(
            Symbol("fnl$i") => FourierNeuralLayer(vol,C,r; activation=activation) 
            for i in 1:n_blocks
        )...,
        outproj =  Conv((1,1),C=>1,use_bias=false)
    ))

    export SpectralConv, FourierNeuralLayer, FNO
end