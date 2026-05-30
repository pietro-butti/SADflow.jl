module Utils
    using ADerrors, FormalSeries
    using LinearAlgebra, StatsBase
    using Lux, ForwardDiff
    using Pipe
    using FFTW, FastTransformsForwardDiff

    dsum(x; dims)  = dropdims(sum(x; dims); dims)
    dmean(x; dims) = dropdims(mean(x; dims); dims)

    sumvol(x::AbstractArray{T,N};dims=nothing) where {T,N} = 
        dsum(x,dims=isnothing(dims) ? Tuple(2:N) : dims)
    batched_sumvol(x::AbstractArray{T,N}) where {T,N} = sumvol(x;dims=Tuple(2:N-1))

    function show_keys(nt::NamedTuple, prefix="", is_last=true)
        ks = collect(keys(nt))
        for (i, key) in enumerate(ks)
            connector = i == length(ks) ? "└─ " : "├─ "
            println(prefix, connector, key)
            
            value = getfield(nt, key)
            if value isa NamedTuple
                extension = i == length(ks) ? "   " : "│  "
                show_keys(value, prefix * extension, i == length(ks))
            end
        end
    end
    export dmean, sumvol, batched_sumvol, show_keys

    batched_fft(x)  =  fft(x,1:(ndims(x)-1))
    batched_ifft(x) = ifft(x,1:(ndims(x)-1))

    stack_complex(x,dim) = cat(real.(x),imag.(x),dims=dim)
    stack_complex(x) = stack_complex(x,ndims(x)-1)

    riffle_complex(x) = begin
        X = stack([real(x),imag(x)])
        X = permutedims(X,circshift(1:ndims(x)+1,1))
        reshape(X,:,size(x)[2:end]...)
    end

    build_source(x,vol) = begin
        (T,V...) = vol
        B = last(size(x))

        x = reshape(x,T,V.÷V...,1,B)
        o = zeros(eltype(x),T,V.-1...,1,B)
        x = cat(x,o,dims=(1:length(V)).+1)
    end

    export batched_fft, batched_ifft, stack_complex, riffle_complex, build_source


    include("Utils_ADerrors.jl")
        export uwcorr

    include("Utils_plotting.jl")
        export meff, plottable

    include("Utils_hutchinson.jl")
        export vjv, vjjv, trJ, trJJ

    include("Utils_lambdaphi4.jl")
        export nn, _action, _grad_S, _Hvp, _fHf, propagator, source

    include("Utils_reweighting.jl")
        export compute_weights, reweight    

end