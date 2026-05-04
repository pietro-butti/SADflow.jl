module SeriesConvs
    using Lux, NNlib, LuxCore, Random, FormalSeries
    
    struct SeriesConv{N} <: Lux.AbstractLuxLayer
        in_chs::Int
        out_chs::Int
        kernel_size::NTuple{N,Int}   # N spatial dims
        pad::Tuple
    end

    # Constructor that auto-computes symmetric padding from kernel size
    function SeriesConv(in_chs, out_chs, kernel_size::NTuple{N,Int}) where N
        pad = ntuple(i -> kernel_size[mod1(i,N)] ÷ 2, 2N)  # symmetric padding for each dim
        SeriesConv{N}(in_chs, out_chs, kernel_size, pad)
    end

    Lux.initialparameters(rng::AbstractRNG, l::SeriesConv) =
        (; weight = randn(rng, Float32, l.kernel_size..., l.in_chs, l.out_chs))
    Lux.initialstates(::AbstractRNG, ::SeriesConv) = (;)

    function (l::SeriesConv)(x::AbstractArray{<:Series{T,O}}, ps, st) where {T,O}
        # do the circular padding
        x = pad_circular(x, l.pad)

        # extract metadata about convolution
        cdims = DenseConvDims(getindex.(x,1), ps.weight; padding=0)

        # broadcast convolution at each order
        coeffs = ntuple(O) do k
            NNlib.conv(getindex.(x,k), ps.weight, cdims)
        end

        # Mount back series object
        y = map(zip(coeffs...)) do s Series(s) end

        return y, st
    end

    function (l::SeriesConv)(x::AbstractArray{<:AbstractFloat}, ps, st)
        cdims = DenseConvDims(x, ps.weight; padding=1)
        return NNlib.conv(x, ps.weight, cdims), st
    end


    export SeriesConv


end