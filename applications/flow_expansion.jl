using Pkg; Pkg.activate(".")
using SADflow
using FormalSeries, ChainRulesCore, Zygote
using Lux, Optimisers, MLUtils
using Random, StatsBase, JLD2, Pipe

## =================== Load pre-generated configurations ===================
    folder   = "/Users/pietro/code/software/SADflow/.local_sketches/"
    ens_name = "2d_l0.05_k0.243_L_8_8" 
    
    data    = load(joinpath(folder,"$ens_name.jld2"))
    vol,λ,κ = data["vol"],data["λ"],data["κ"]

    ϕ = data["ϕ"] .|> Float64
    
    D = length(vol)
    cfgs = eachslice(ϕ,dims=ndims(ϕ))
##
    roll(func) = phi -> begin
        N = ndims(phi)
        confs = @pipe selectdim(phi,N-1,1) |> eachslice(_,dims=N-1)
        map(func,confs)    
    end
    action = _action(λ,κ) |> roll

    srcobs = source |> roll
## ==========================================================================


## === Helpers and net =====================================================
    const T0 = Float64                  
    mkε()  = Series((zero(T0), one(T0),  zero(T0)))
    mk1()  = Series((one(T0),  zero(T0), zero(T0)))
    mk0()  = Series((zero(T0), zero(T0), zero(T0)))
    const ε = mkε()

    coeffs(s, k)  = getindex(s, k)
    zero_ldj(n) = fill(mk0(), n)
    promote0(f) = mk1() .* f

    function make_net(T0)
        Parallel(
            nothing,
            Chain(
                SeriesConv(1, 1, (3, 3)),
                SeriesConv(1, 1, (3, 3)),
                SeriesConv(1, 1, (3, 3)),
                SeriesConv(1, 1, (3, 3)),
                SeriesConv(1, 1, (3, 3)),
                SeriesConv(1, 1, (3, 3)),
                WrappedFunction(x -> ε .* x)
            ),                              # ε·s
            WrappedFunction(x -> zero(T0))  # t = 0
        )                                 
    end
## --- Net definition -------------
    rng = Xoshiro(1994)
    net = make_net(Float64)

    model = AffineEOCoupling(net,(vol...,1),true)
    ps, st = Lux.setup(rng,model)
## --- Loss function ---------------
    function inverseKL(model, N_layers, ps, st, zin)
        ldj = zero_ldj(last(size(zin)))
        (zout,logdetJ),st = model((zin,ldj), N_layers, ps, st);

        logr = - action(zin)
        logp = - action(zout) .+ ε .* srcobs(zout)
        logq = logr .- logdetJ

        logw = logp .- logq

        ESS = 1 / ( 1 + var(coeffs.(logw,2)))
        KL₂ = coeffs.(-logw,3) |> mean

        return KL₂, st, (; ess=ESS)
    end
## =========================================================================


## === TRAINING ============================================================
    # Metaparameters -----------
        N_LAYERS = 10 
        η        = 0.1
        BSIZE    = 100
        SAMPLES_PER_EPOCH = 1000
    # --------------------------
    ts = Training.TrainState(cl, ps, st, Adam(η));
    Nconf = last(size(ϕ))
    ϕtrain = ϕ
    ϕ̃ = promote0(ϕtrain)
## --- Pre-compilation -------------------------------------
    zin = ϕ[:,:,:,1:BSIZE] |> promote0
    @time l,_,metr = inverseKL(model,N_LAYERS,ps,st,zin)

    _, loss, metr, ts = @time Training.single_train_step!( 
        AutoZygote(),  
        inverseKL, zin, ts;
    );
## --- Training ---------------------------------------------
    metrics = DataFrame((loss=[], ess=[]))

    for epoch in 1:700
        stime = time()

        for i in 1:SAMPLES_PER_EPOCH÷BSIZE
            i_samples = sample(rng,1:Nconf,BSIZE)
            ϕ_sample = ϕtrain[:,:,:,i_samples] |> promote0

            _, loss, mtr, ts = Training.single_train_step!( 
                AutoZygote(),  
                inverseKL, ϕ_sample, ts;
            );
        end
        ttime = time()-stime

        # Training logs ----------
            (_ps,_st) = ts.parameters, Lux.testmode(ts.states)
            l,_,m   = inverseKL(model,N_LAYERS,_ps,_st,ϕ̃[:,:,:,sample(rng,1:Nconf,SAMPLES_PER_EPOCH)])

            push!(metrics,(;loss=l, ess=m.ess))

            @printf("epoch=%i [%.3f s]: loss=%.7f ess=%.7f \n",epoch, ttime, l, m.ess)
    end

## ==========================================================================================

using Plots

plot(metrics.loss,yscale=:log10,yaxis="loss",label="loss")
plot!(twinx(),1 .- metrics.ess,yscale=:log10,yaxis="ESS",label="ESS",color="red")
