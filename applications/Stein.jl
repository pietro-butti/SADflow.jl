using Pkg; Pkg.activate(".")
using Revise
using LinearAlgebra, Random, StatsBase
using Plots, Pipe, Printf, DataFrames, Colors, JLD2
using Lux, NNlib, MLUtils, Optimisers
using Zygote, ForwardDiff
using FormalSeries, ADerrors
using Lattice, SADflow

## ======================================================
    folder   = "/Users/pietro/code/software/SADflow/.local_sketches/"
    ens_name = "2d_l0.0_k0.12461_L_8_8" 
    
    data = load(joinpath(folder,"$ens_name.jld2"))

    vol,λ,κ = data["vol"],data["λ"],data["κ"]
    ϕ = data["ϕ"]
    
    ## Float32 conversion
    λ,κ = Float32(λ),Float32(κ)
    ϕ = Float32.(ϕ)

    cfgs = eachslice(ϕ,dims=ndims(ϕ))
##
    f_free = free_propagator(vol,κ,λ)

    fstar_norm = dsum(f_free,dims=2) 
    fstar_norm ./= first(fstar_norm)
    
    Δf_t = fout -> begin
        c      = dsum(fout,dims=(2,3))
        c_norm = map(eachcol(c)) do x x./first(x) end |> stack
        abs.(c_norm .- fstar_norm) ./ fstar_norm
    end

## =======================================================


## =============================================================
## MLP
    V = prod(vol)
    net(vol) = Chain(
        FlattenLayer(),
        Dense(V =>V),
        # Dense(V =>32),
        # Dense(32=>32),
        # Dense(32=>32),
        # Dense(32=>V),
        ReshapeLayer((vol...,1)),
    )
    model = net(vol)
##
    seed  = 1994
    rng = Xoshiro(seed)
    ps, st = Lux.setup(rng, model)
##
    div_f2(model,ps,st,z) = begin
        J = map(reverse(keys(ps))) do k
            if haskey(ps[k],:weight)
                ps[k].weight
            else
                1
            end
        end |> prod

        divf2 = tr(J*J)
        return [divf2 for _ in eachslice(z,dims=ndims(z))]
    end
## ==============================================================


## ---------------------------- TRAINING --------------------------
    # Metaparameters -----------
        NSRC     = 1
        η        = 0.001f0
        λreg     = 0.01f0
        BSIZE    = 50
        NSAMPLES = 1000
    # --------------------------

    # LOSS = (args...) -> KL2(args...,λ,κ; ns=NSRC, rng=rng, div_f2=div_f2)
    LOSS = (args...) -> KL2(args...,λ,κ; ns=NSRC, rng=rng, div_f2=nothing)

    # opt = Adam(η)
    opt = OptimiserChain(WeightDecay(λreg),Adam(η))

    ts = Training.TrainState(model, ps, st, opt);
    ϕtrain = ϕ[:,:,:,1:NSAMPLES]
    ϕtest  = ϕ[:,:,:,901:1000] 
## Pre-compilation
    zin = ϕ[:,:,:,1:BSIZE]
    @time l,_,metr = LOSS(model,ps,st,zin)

    _, loss, metr, ts = @time Training.single_train_step!( 
        AutoZygote(),  
        LOSS, zin, ts;
    );

##
    mm = NamedTuple{keys(metr)}([mean(i)] for i in values(metr))
    metrics = DataFrame(
        (loss=[l], loss_test=[0.], deltaf=[1.], mm...)
        )
## training
    σ² = var(Obs(ϕ))

   for epoch in 1:500
        stime = time()
        for i in 1:NSAMPLES÷BSIZE
            i_samples = sample(rng,1:NSAMPLES,BSIZE,ordered=true)
            ϕ_sample = ϕtrain[:,:,:,i_samples]

            _, loss, mtr, ts = Training.single_train_step!( 
                AutoZygote(),  
                LOSS, ϕ_sample, ts;
            );
        end
        ttime = time()-stime

        # Training logs ----------
            (_ps,_st) = ts.parameters, Lux.testmode(ts.states)
            l,_,m   = LOSS(model,_ps,_st,ϕtrain)
            lt,_,mt = LOSS(model,_ps,_st,ϕtest)
            mm = NamedTuple{keys(m)}(mean.(values(m)))

            fout,_ = model(ϕtrain,_ps,_st)

            Dm = maximum(Δf_t(fout),dims=2) |> mean

            _l = (loss=l, loss_test=lt, deltaf=Dm, mm...)

            push!(metrics,_l)

            @printf("epoch=%i [%.3f s]: %.7f  %.7f\n",epoch, ttime, l+σ², Dm)
        
        Dm < 1e-0 ? break : continue
    end
## ---------------------------------------------------------------------


## =====================================================================
    fϕ,_ = model(ϕ,ts.parameters,ts.states)

    fnorm = fϕ[:,:,1,372]
    fnorm = map(eachcol(fnorm)) do y y./first(y) end |> stack

    f0 = f_free ./ first(f_free)

    plt2 = heatmap(f0)
    plt3 = heatmap(fnorm)
    plt4 = heatmap(abs.(fnorm .- f0)./f0 .* 100)

    plot(plt2,plt3,plt4,layout=(1,3),size=(600,200))
## ---------------------------------------------------------------------
    # maximum(Δf_t(fϕ))
    corr = dsum(fϕ,dims=(2,3))
    corr = stack(map(eachcol(corr)) do x x./first(x) end)

    c = map(eachrow(corr)) do c uwreal(Float64.(c[:]),ens_name) end
    uwerr.(c)

    y,ye = value.(c), ADerrors.err.(c)

    p1 = plot()

    f0i = sum(f_free,dims=2)
    f0i ./= first(f0i)

    scatter!(p1,y,yerror=ye,ylim=(minimum(y)*0.7,Inf),yscale=:log10)
    scatter!(p1,axes(f0i),f0i,ms=2,yscale=:log10 )

    p2 = scatter(maximum(Δf_t(fϕ),dims=2))

    p = plot(p1,p2,linx=:x,layout=(2,1))
## ---------------------------------------------------------------------
    
    yl = abs.(metrics.loss .+ σ²)
    yt = abs.(metrics.loss_test .+ σ²)

    ylims=(-Inf,Inf)
    # ylims=(minimum(yl[100:end])*0.95,maximum(yl[100:end])*1.05) 

    plt1 = plot(yscale=:log10,ylim=ylims,ylabel="abs(loss)")
    plot!(plt1,yl,label="train")
    plot!(plt1,yt,label="test")

    plt2 = plot(yscale=:log10)
    plot!(plt2,metrics.deltaf, label="<∑(|f_true - f_learned|/f_true)²>")

    p = plot(plt1,plt2,layout=(2,1),size=(500,500),link=:x)

## ---------------------------------------------------------------------

    import Base: length, iterate
    length(x::uwreal) = 1
    iterate(x::uwreal) = (x, nothing)
    iterate(x::uwreal,::Nothing) = nothing

    action = ϕ ->
        ϕ.^2 .+ λ .* (ϕ.^2 .- 1).^2 .+
        -2κ .* ϕ .* (
            circshift(ϕ,(-1,0)) .+ circshift(ϕ,(0,-1))
        ) |> sum
    obs(z; x₀=1) = dsum(z,dims=2)[x₀] 

    z = deepcopy(ϕ)
    
    # 2pt function
        corr2 = dsum(ϕ[1:1,:,:,:] .* ϕ,dims=(2,3))
        corr2 = map(eachrow(corr2)) do x uwreal(Float64.(x),ens_name) end 
        
        corr2 ./= first(corr2)
        uwerr.(corr2)

    # Exact reweighting
        c2 = dsum(z .+ ε .* f0i,dims=(2,3)) 
        corrw = map(eachrow(c2)) do r uwreal(Float64.(getindex.(r,2)[:]),ens_name) end
        corrw ./= first(corrw)
        uwerr.(corrw)

    # Improved reweigthing
        func = StatefulLuxLayer(model,ts.parameters,ts.states)
        ε = Series((0.f0,1.f0))

        Jac = batched_jacobian(func, AutoForwardDiff(), ϕ)
        trj = tr.(eachslice(Jac,dims=3))

        w = compute_weight(ϕ,func(ϕ),action,obs,trj)
        corri = reweight_corr(w,ϕ .+ ε.*func(ϕ),ens_name)
        corri ./= first(corri)

        uwerr.(corri)
##
    plt = plot(yscale=:log10,ylim=(1e-12,100))

        scatter!(plt,value.(corr2),yerror=ADerrors.err.(corr2),label="2pts")
        plot!(plt,value.(corrw),label="exact")
        scatter!(plt,value.(corri),yerror=ADerrors.err.(corri),label="Improved reweighting")
    
    display(plt)

