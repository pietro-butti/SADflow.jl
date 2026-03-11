using Pkg; Pkg.activate(".")
using SADflow, Revise
using LinearAlgebra, Random, StatsBase
using Lux, NNlib, MLUtils, Optimisers
using Zygote, ForwardDiff
using Plots, Pipe, Printf, DataFrames, Colors, JLD2
using FormalSeries, Lattice
using ADerrors
using FFTW, FastTransformsForwardDiff

## ======================================================
    folder   = "/Users/pietro/code/software/SADflow/.local_sketches/"
    ens_name = "2d_l0.0_k0.12461_L_32_8" 
    
    data = load(joinpath(folder,"$ens_name.jld2"))

    vol,λ,κ = data["vol"],data["λ"],data["κ"]
    ϕ = data["ϕ"]
    
    ## Float32 conversion
    λ,κ = Float32(λ),Float32(κ)
    ϕ = Float32.(ϕ)

## =======================================================


## =============================================================
## MLP
    net(vol) = Chain(
        FlattenLayer(),
        Dense(prod(vol)=>prod(vol),tanh),
        ReshapeLayer((vol...,1))
    )
    model = net(vol)
##
    seed  = 1994
    rng = Xoshiro(seed)
    ps, st = Lux.setup(rng, model)
## ==============================================================




## ---------------------------- TRAINING --------------------------
    # Metaparameters -----------
        NSRC     = 1
        η        = 0.005f0
        BSIZE    = 50
        NSAMPLES = 1000
    # --------------------------

    LOSS = (args...) -> KL2(args...,λ,κ; ns=NSRC)

    ts = Training.TrainState(model, ps, st, Optimisers.Adam(η));
    ϕtrain = ϕ[:,:,:,1:NSAMPLES]
    ϕtest  = ϕ[:,:,:,1001:2000] 
## Pre-compilation
    zin = ϕ[:,:,:,1:BSIZE]
    @time l,_,metr = LOSS(model,ps,st,zin)

    _, loss, metr, ts = @time Training.single_train_step!( 
        AutoZygote(),  
        LOSS, zin, ts;
    );

##
    metrics = DataFrame(
        loss = [], loss_test = [], deltaf = [],
        trJJ=[], fHf=[], fDO=[]
    )
## training
   for epoch in 1:500
        stime = time()
        for i in 1:NSAMPLES÷BSIZE
            i_samples = sample(rng,1:NSAMPLES,BSIZE,ordered=true)
            ϕ_sample = ϕtrain[:,:,:,i_samples]

            _, loss, mtr, ts = Training.single_train_step!( 
                AutoZygote(),  
                LOSS, ϕ_sample, ts;
            );

            # @printf("step=%i [%.3f s]: %.7f  %.7f\n",epoch, ttime, l, Dm)

        end
        ttime = time()-stime

        # Training logs ----------
            (_ps,_st) = ts.parameters, Lux.testmode(ts.states)
            l,_,m   = LOSS(model,ps,st,ϕtrain)
            lt,_,mt = LOSS(model,ps,st,ϕtest)
            mm = NamedTuple{keys(m)}(mean.(values(m)))

            fout,_ = model(ϕtrain,_ps,_st)
            # Dm = eachslice(1 .- f_true[:,:,1,1] ./ fout,dims=4) .|> norm |> mean            
            Dm = 1000

            push!(metrics,(l,lt,Dm,mm...))

            # @printf("epoch=%i [%.3f s]: %.7f  %.7f\n",epoch, ttime, l, Dm)
            @printf("epoch=%i [%.3f s]: %.7f  %.7f\n",epoch, ttime, l, mm.trJJ)
    end
## ---------------------------------------------------------------------