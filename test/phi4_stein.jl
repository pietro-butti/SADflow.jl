using Pkg; Pkg.activate(".")
using SADflow, Revise
using FormalSeries, Pipe
using Plots, Random, StatsBase
using Lux, NNlib, MLUtils, Optimisers
using Zygote
using LinearAlgebra


## ============== Read configuration from ensemble ==============
    using Lattice
    
    # Define utils --------------------------------------------
        k = 0.3
        L = 16
    
        LOCATION = "/Users/pietro/code/julia/sketches/inverseRG/CONFS0/"
        ens_name = "2d_l1.145_k$(k)_L$L"

        function store_album(ws,space)
            return (prefix,suffix,itr) -> begin
                pics = Array{Float32}(undef, space.iL...,1,0)
                for icfg in itr
                    Lattice.Fields.read!(ws._phi, "$prefix$(suffix(icfg))")
                    pics = cat(pics,conf_to_pic(ws._phi,space),dims=ndims(pics))
                end
                pics
            end
        end
    # Infer configuration indexes -----------------------------
        folder = joinpath(LOCATION,ens_name)
        conflist = filter(x->endswith(x,".jld2"),readdir(folder))
        str_idx = map(x->split(x,".")[end-1][4:end], conflist)
        confidx = parse.(Int,str_idx)
    # Setup readers -------------------------------------------
        pars  = Phi4_params(k,0.145,1.)
        space = Grid{2}((L,L),(8,8))
        ws    = Phi4_workspace(space)
        
        store_func = store_album(ws, space)
        prefix = "$folder/$ens_name"
        suffix = i -> ".cfg$(lpad(i,5,'0')).jld2"
        pics = store_func("$folder/$ens_name", suffix, confidx)
    # ---------------------------------------------------------
## ===============================================================

## ============= Function to compute the action =================
    neighbor_sum(ϕ::AbstractArray{T,2}) where T = 
        circshift(ϕ,(0,1)) .+ 
        circshift(ϕ,(1,0))

    neighbor_sum(ϕ::AbstractArray{T,4}) where T = 
        circshift(ϕ,(1,0,0,0)) .+ 
        circshift(ϕ,(0,1,0,0)) .+ 
        circshift(ϕ,(0,0,1,0)) .+ 
        circshift(ϕ,(0,0,0,1)) 

    Actionλϕ⁴(ϕ::AbstractArray{T,N},λ::T,κ::T) where {T,N} = (
        ϕ.^2 .+ λ .* (ϕ.^2 .- one(T)).^2 .+
        -2κ .* ϕ .* neighbor_sum(ϕ)
    ) |> x->sum(x; dims=1:N-1)

    staple_sum(ϕ::AbstractArray{T,2}) where T = 
        circshift(ϕ,(0,1)) .+ circshift(ϕ,(0,-1)) .+ 
        circshift(ϕ,(1,0)) .+ circshift(ϕ,(-1,0))

    staple_sum(ϕ::AbstractArray{T,4}) where T = 
        circshift(ϕ,(1,0,0,0)) .+ circshift(ϕ,(-1,0,0,0)) .+ 
        circshift(ϕ,(0,1,0,0)) .+ circshift(ϕ,(0,-1,0,0)) .+ 
        circshift(ϕ,(0,0,1,0)) .+ circshift(ϕ,(0,0,-1,0)) .+ 
        circshift(ϕ,(0,0,0,1)) .+ circshift(ϕ,(0,0,0,-1)) 


    function Forceλϕ⁴(ϕ::AbstractArray{T,N},λ::T,κ::T) where {T,N}
        two = one(T)+one(T)
        -2κ .* staple_sum(ϕ) .+ two.*ϕ + two*two*λ .* (ϕ.^2 .- one(T)) .* ϕ
    end
    # ) |> x->sum(x; dims=1:N-1)

    fHf(ϕ::AbstractArray{T,N},f::AbstractArray{T,N},λ::T,κ::T) where {T,N} = (
        -4κ .* f .* staple_sum(f) .+ 
        (one(T)+one(T) - 4λ .+ 12λ .* ϕ .* ϕ) .* f .* f
    ) |> x->sum(x; dims=1:N-1)

## ===============================================================



## ========================= Define model =========================
    vol = (16,16)
    (λ,κ)  = (1.145f0, 0.3f0)
    seed = 1994

    ϕ = Float32.(pics)
    
    rng = Xoshiro(seed)

    pad = WrappedFunction(x->pad_circular(x,(2,0,2,0)))
    net() = Chain(
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1, leakyrelu; use_bias=false),
        pad,Conv((3,3),1=>2, tanh; use_bias=false),
        x->biject(x; dim=3)
    )

##
    model = Chain(
        AffineEOCoupling(net(),vol,true),
        AffineEOCoupling(net(),vol,false),
        AffineEOCoupling(net(),vol,true),
        AffineEOCoupling(net(),vol,false),
        AffineEOCoupling(net(),vol,true),
        AffineEOCoupling(net(),vol,false),
        AffineEOCoupling(net(),vol,true),
        AffineEOCoupling(net(),vol,false)
    )

    ps, st = Lux.setup(rng, model)
##

## =================== Define loss function ======================
    trace_J2 = (func,z) -> begin
        η = rand([-1,1],size(z)...)
        v = jacobian_vector_product(func, AutoForwardDiff(), z, η)
        w = jacobian_vector_product(func, AutoForwardDiff(), z, v)
        sum(η .* w, dims=(1,2))[:]
    end

    function KLsecond(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T) where {T,N}
        func = StatefulLuxLayer(model,ps,st)

        # J = Lux.batched_jacobian(func, AutoForwardDiff(), ϕ)
        # trJ² = map(j -> tr(j*j),eachslice(J,dims=3))[:]
        
        trJ² = mean(stack([trace_J2(func,ϕ) for _ in 1:100]),dims=2)[:]

        f = func(ϕ)
        f⊥Hf = fHf(f,ϕ,λ,κ)[:]
        f_av = map(mean, eachslice(f,dims=4))

        return mean(trJ² + f⊥Hf + f_av), st, (;)
    end
    lossKL2(λ,κ) = (m,p,s,z) -> KLsecond(m,p,s,z,λ,κ)
##
    func = StatefulLuxLayer(model,ps,st)
    
    ϕ̃ = ϕ[:,:,:,1:12]
    
    J = Lux.batched_jacobian(func, AutoForwardDiff(), ϕ̃)
    t = map(j -> tr(j*j),eachslice(J,dims=3))[:]

    trs = []
    for is in 1:1000
        η = rand([-1,1],size(ϕ̃)...)
        v = jacobian_vector_product(func, AutoForwardDiff(), ϕ̃, η)
        w = jacobian_vector_product(func, AutoForwardDiff(), ϕ̃, v)
        push!(trs,sum(η .* w,dims=(1,2))[:])
    end


    mt = [mean(trs[1:i]) for i in 1:1000]
    
    plt = plot()
    plot!(plt,mt)
    hline!(plt,t)
## =================================================================

## ========================== Training =============================
    LOSS = lossKL2(λ,κ)
    η = 0.001f0
    BSIZE = 50
    NSAMPLES = size(ϕ,ndims(ϕ))

    metrics = (hloss=[],ess=[])
    ts = Training.TrainState(model, ps, st, Optimisers.Adam(η));

##    
    lossh = []
    for step in 1:100
        stime = time()

        # Proper training step
        i_samples = sample(rng,1:NSAMPLES,BSIZE,ordered=true)
        ϕ_sample = ϕ[:,:,:,i_samples]
        _, loss, _, ts = Training.single_train_step!( 
            AutoZygote(),  
            LOSS, ϕ_sample, ts;
        );

        # Logs
        ttime = round(time()-stime, digits=1)
        @show step, ttime, loss
        push!(lossh,loss)
    end
##
plot(lossh,yscale=:log10)
##

    using JLD2
    ps_trained = deepcopy(ts.parameters)
    st_trained = deepcopy(ts.states)
    @save "trained_model.jld2" ps_trained st_trained

    # @load "trained_model.jld2" ps_trained st_trained

