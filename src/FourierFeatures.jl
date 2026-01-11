module FourierFeatures
    using Lux, NNlib
    using LuxCore, Random, WeightInitializers

    ## ======================== Lattice Convolution ========================== ##
        
        # -------------- Stencil definition -------------
            struct Stencil{K}
                w::Vector{Matrix}
            end
            Stencil(::Val{3}) = Stencil{3}([
                [[0 0 0]; [0 1 0]; [0 0 0]],
                [[0 1 0]; [1 0 1]; [0 1 0]],
                [[1 0 1]; [0 0 0]; [1 0 1]]
            ])
            Stencil(::Val{5}) = Stencil{5}([
                [[0 0 0 0 0]; [0 0 0 0 0]; [0 0 1 0 0]; [0 0 0 0 0]; [0 0 0 0 0]],
                [[0 0 0 0 0]; [0 0 1 0 0]; [0 1 0 1 0]; [0 0 1 0 0]; [0 0 0 0 0]],
                [[0 0 1 0 0]; [0 0 0 0 0]; [1 0 0 0 1]; [0 0 0 0 0]; [0 0 1 0 0]],
                [[0 0 0 0 0]; [0 1 0 1 0]; [0 0 0 0 0]; [0 1 0 1 0]; [0 0 0 0 0]],
                [[1 0 0 0 1]; [0 0 0 0 0]; [0 0 0 0 0]; [0 0 0 0 0]; [1 0 0 0 1]],
                [[0 1 0 1 0]; [1 0 0 0 1]; [0 0 0 0 0]; [1 0 0 0 1]; [0 1 0 1 0]]
            ])
            Stencil(kdim::Int) = Stencil(Val(kdim))

            n_pars(kdim) = begin
                n = Int(ceil(kdim/2))
                Int(n * (n+1) / 2)
            end 
        # ------------------------------------------------

        struct LatConv2D{F1} <: LuxCore.AbstractLuxLayer
            kernel_dim::Int
            init_weight::F1
        end
        LatConv2D(k_dim::Int; init_weight=zeros32) = begin
            iseven(k_dim) ? 
            error("LatConv2D only implemented for odd dimensions") : 
            LatConv2D{typeof(init_weight)}(k_dim,init_weight)
        end

        LuxCore.initialparameters(rng::AbstractRNG, l::LatConv2D) = (;
            weight=l.init_weight(rng, n_pars(l.kernel_dim)),
        )
        LuxCore.initialstates(::AbstractRNG, l::LatConv2D)  = (;
            stencil = Stencil(l.kernel_dim)
        )

        # --------------- Forward pass ----------------
        function (l::LatConv2D)(x::AbstractArray{N,T}, ps, st::NamedTuple) where {N,T}
            kernel = sum(ps.weight .* st.stencil.w)
            kernel = reshape(kernel, size(kernel)...,1,1)

            z = NNlib.pad_circular(x,(2,0,2,0))   # only valid for 2 dimensions with kernelsize=3
            z = NNlib.conv(z,kernel)              

            return z, st
        end
        # ---------------------------------------------
    ## ======================================================================= ##

    ## ========================= Fourier Feature Model =========================
        struct FourierFeature{C,F1} <: LuxCore.AbstractLuxContainerLayer{(:conv,)}
            conv::C
            init_freq::F1
        end
        FourierFeature(n::Int; init_freq=glorot_normal) = FourierFeature(LatConv2D(n), init_freq)

        LuxCore.initialparameters(rng::AbstractRNG, f::FourierFeature) = (;
            conv = LuxCore.initialparameters(rng,f.conv),
            freq = f.init_freq(rng,1)  
        )
        LuxCore.initialstates(rng::AbstractRNG, f::FourierFeature) = (;
            conv = LuxCore.initialstates(rng, f.conv),
            rng,    
        )

        # -------------- Forward pass --------------
        #       ϕ -> k ⋆ sin.(ω .* ϕ)
        function (f::FourierFeature)(x,ps,st)
            z = sin.(only(ps.freq) .* x)
            z,st_conv = f.conv(x,ps.conv, st.conv)
            return z, merge(st, (; conv=st_conv))
        end
        # -----------------------------------------


        FourierConv(n_ch::Int,ch_dim::Int) = BranchLayer(
            [FourierFeature(ch_dim) for _ in 1:n_ch]...
            ; fusion= (xs...) -> reduce(+,xs)
        )

    ## ===========================================================================

    export LatConv2D, FourierFeature, FourierLatConv

end