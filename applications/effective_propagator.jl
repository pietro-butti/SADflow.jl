using Pkg; Pkg.activate(".")
using Revise 
using Random, StatsBase, Plots, LinearAlgebra, Pipe, DataFrames, JLD2, Printf, LaTeXStrings, Colors
using Lux, Optimisers, Zygote
using Lattice, SADflow
using ADerrors, FormalSeries

jc = Colors.JULIA_LOGO_COLORS

import ADerrors.err

## =================== Load pre-generated configurations ===================
    folder   = "/Users/pietro/code/software/SADflow/.local_sketches/"
    ens_name = "2d_l0.05_k0.243_L_32_8" 
    
    data    = load(joinpath(folder,"$ens_name.jld2"))
    vol,λ,κ = data["vol"],data["λ"],data["κ"]

    ϕ = data["ϕ"]
    
    D = length(vol)
    cfgs = eachslice(ϕ,dims=ndims(ϕ))

    uwc(x) = uwcorr(x,ens_name)
    uw(x) = uwreal(x,ens_name)
##
    ziproll(func) = (phi,f) -> begin
        @assert ndims(phi)==ndims(f)
        N = ndims(phi)
        confs = @pipe selectdim(phi,N-1,1) |> eachslice(_,dims=N-1)
        vs    = @pipe selectdim(f,N-1,1)   |> eachslice(_,dims=N-1)
        map(zip(confs,vs)) do (c,v) func(c,v) end
    end

    roll(func) = phi -> begin
        N = ndims(phi)
        confs = @pipe selectdim(phi,N-1,1) |> eachslice(_,dims=N-1)
        map(func,confs)    
    end

    Hvp = ziproll(_Hvp(λ,κ))
    action = _action(λ,κ)

    m²(κ,λ,vol) = (1-2λ)/2κ - 2*length(vol)
    f_free = propagator(κ,1/κ - 2D, vol)

    Δf_t = fout -> begin
        Gt0 = sumvol(f_free)
        Gt  = batched_sumvol(fout)
        abs.(Gt .- Gt0) ./ Gt0
    end
## ==========================================================================


## =============================================================
    function EffectivePropagator(vol)
        V = prod(vol)
        Nₜ = first(vol)
        return @compact(;
            Znet = Dense(2V=>2,softplus),
            Rnet = Dense(2V=>1,tanh),
        ) do x
            @assert ndims(x)==2
            @assert size(x,1)==2V
            
            Z,Σ = x |> Znet |> SADflow.biject
            R   = x |> Rnet
            
            T = eltype(x) 
            p₀ = T(2π/Nₜ) .* T.((1:Nₜ).-1)
            p̂₀ = T(2) .* sin.(p₀ ./ T(2))

            @return Z ./ (p̂₀.^2 .+ Σ ) .+ R
        end
    end
##
    net = (vol,κ) -> begin
        Nₜ = first(vol)
        Nₓ = prod(vol[2:end])
        factor = Nₓ/2κ

        Chain((
            fft            = WrappedFunction(batched_ifft),
            effective_prop = Chain((
                complex_policy = WrappedFunction(riffle_complex), # alternatives: riffle, real, stack+conv
                flatten        = FlattenLayer(),
                propagator     = EffectivePropagator(vol),  
            )),
            # mode_mixing    = Chain(Dense(Nₜ=>8),Dense(8=>Nₜ)),
            mode_mixing    = Dense(Nₜ=>Nₜ),
            build_source   = WrappedFunction(x->build_source(x,vol)),
            ifft           = WrappedFunction(batched_fft),
            normalization  = WrappedFunction(x->x*factor),
            complex_proj   = Chain(
                WrappedFunction(stack_complex),
                Conv((1,1),2=>1,use_bias=false)
            )
        ))
    end

    

    model = net(vol,κ)
    seed  = 1994
    rng = Xoshiro(seed)
    ps, st = Lux.setup(rng, model)

    ps = f64(ps)
    st = f64(st)
## ==============================================================




## ---------------------------- TRAINING --------------------------
    fHf(ϕ,f) = @pipe stack(Hvp(ϕ,f)) .* selectdim(f,ndims(ϕ)-1,1) 


    function KL2(model,ps,st,z::AbstractArray{T,N}; ns=1, rng=Random.default_rng(),) where {T,N} 
        func = StatefulLuxLayer(model,ps,st)
        f = func(z)

        trJ² = trJJ(rng, func, z; ns=ns)
        f_Hf = dsum(fHf(z,f),dims=Tuple(1:N-2))
        f∇O  = eachslice(f,dims=N) .|> source 
        
        return mean(trJ² .+ f_Hf .- 2 .* f∇O), st, (; trJ², f_Hf, f∇O)
    end  

##
    # Metaparameters -----------
        NSRC     = 1
        η        = 0.01
        λreg     = 0.
        BSIZE    = 100
        SAMPLES_PER_EPOCH = 1000
    # --------------------------

    LOSS = (args...) -> KL2(args...; ns=NSRC, rng=rng)

    opt = OptimiserChain(WeightDecay(λreg),Adam(η))

    ts = Training.TrainState(model, ps, st, opt);

    Nconf = last(size(ϕ))
    ϕtrain, ϕtest = stack(cfgs[1:Nconf-SAMPLES_PER_EPOCH]), stack(cfgs[Nconf-SAMPLES_PER_EPOCH:end])
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
    σ² = var(source.(eachslice(ϕ,dims=4)))
    Nconf = last(size(ϕtrain))

   for epoch in 1:500
        stime = time()

        for i in 1:SAMPLES_PER_EPOCH÷BSIZE
            i_samples = sample(rng,1:Nconf,BSIZE)
            ϕ_sample = ϕtrain[:,:,:,i_samples]

            _, loss, mtr, ts = Training.single_train_step!( 
                AutoZygote(),  
                LOSS, ϕ_sample, ts;
            );
        end
        ttime = time()-stime

        # Training logs ----------
            (_ps,_st) = ts.parameters, Lux.testmode(ts.states)

            phi = cfgs[sample(rng,last(axes(ϕtrain)),SAMPLES_PER_EPOCH)] |> stack

            l,_,m   = LOSS(model,_ps,_st,phi)
            lt,_,mt = LOSS(model,_ps,_st,ϕtest)
            mm = NamedTuple{keys(m)}(mean.(values(m)))

            fout,_ = model(ϕtrain,_ps,_st)

            Dm = maximum(Δf_t(fout),dims=2) |> maximum

            _l = (loss=l, loss_test=lt, deltaf=Dm, mm...)

            push!(metrics,_l)

            @printf("epoch=%i [%.3f s]: %.7f  %.7f\n",epoch, ttime, l+σ², Dm)
        
        # # Dm < 1e-0 ? break : continue
    end
## ---------------------------------------------------------------------






## --- Propagator -----------------------------------
    ff = propagator(1/κ - 2D,vol)

    func = StatefulLuxLayer(model,ts.parameters,ts.states)
    fout = Float64.(func(ϕ))

##
    m2 = round(m²(κ,λ,D),digits=4)
    lam = λ*(2κ)^2

    off = 1e-12

    p = plot(
        formatter=:latex,framestyle=:box,
        size=(1200,800),
        yscale=:log10,
        guidefontsize=30, 
        tickfontsize=20, 
        legendfontsize=15,
        titlefontsize=30,
        legend=:bottomright,
        title=L"L=%$(first(vol))\times %$(last(vol)),\, \hat\lambda=%$lam,\, \hat m^2=%$m2",
        # xlabel=L"x_0/a",
        ylabel=L"G(x_0)",
        bottom_margin=10Plots.mm,
        left_margin=10Plots.mm,
        right_margin=10Plots.mm,
        top_margin=10Plots.mm,
    )

    l = "analytical"
    f0prop = dsum(ff,dims=2)
    scatter!(p,f0prop,m=:square,c=jc.blue,msc=:auto,label=l,ms=10)
    
    t = 0:length(f0prop)-1
    l = "fk"
    y = uwc(dsum(fout,dims=Tuple(2:D+1))); uwerr.(y)
    d,o = plottable(y,off=off)
    scatter!(p,t.+1,d;o...,label="NN",c=jc.red,msc=jc.red,ms=10,m=:diamond)

    display(p)
##
    p2 = plot(
        formatter=:latex,framestyle=:box,
        size=(1200,400),
        # yscale=:log10,
        guidefontsize=30,
        tickfontsize=20, 
        legendfontsize=15,
        # xlabel=L"x_0/a",
        # ylabel=L"G(x_0)",
        ylabel=L"\Delta",
        xlabel=L"x_0/a",
        bottom_margin=10Plots.mm,
        left_margin=10Plots.mm,
        right_margin=10Plots.mm,
        # ylims=(-1e-10,1e-10)
    )

    d = abs.(y .- f0prop)./f0prop; uwerr.(d)
    scatter!(p2,value.(d),yerror=err.(d),c=jc.green,msc=jc.green,ms=10,label="analytical - FK")


    plt = plot(p,p2,layout=(2,1),size=(1000,1000))
    display(plt)
    # savefig(plt,joinpath(PATH,"PROPAGATOR_NN_$(ens_name).pdf"))
## ---------------------------------------------------




## --- Correlator ------------------------------------    
    ff = propagator(m²(κ,λ,D),vol)

    # --- 2pt function 
        x = sumvol(ϕ,dims=Tuple(2:D))[:,1,:]
        corr2 = Float64.(x[1:1,:] .* x) |> uwc
        corr2 ./= first(corr2)
        uwerr.(corr2)

    # Exact reweighting
        # ε = Series((0.,1.))
        # c2 = dsum(ϕ .+ ε .* ff,dims=(2,3)) 
        # corrw = map(eachrow(c2)) do r uwreal(Float64.(getindex.(r,2)[:]),ens_name) end
        # corrw ./= first(corrw)
        # uwerr.(corrw)

        M2 = m²(κ,0.,D) + 0.5
        ff = propagator(M2,vol)

        ε = Series((0.,1.))
        pt = dsum(ϕ .+ ε .* ff,dims=D+1); 
        new = eachslice(pt,dims=ndims(pt));
        w = -(action.(new) .- ε .* source.(new)) .+ action.(selectdim.(cfgs,3,1)) .|> exp;
        w̃ = uw(w)

        y = reshape(w,1,:) .* dsum(pt,dims=Tuple(2:D))
        corrw = uwcorr(y,ens_name) ./ w̃

        corrw = getindex.(corrw,2)
        corrw ./= first(corrw)
        uwerr.(corrw)


    # Improved reweigthing
        func = StatefulLuxLayer(model,ts.parameters,ts.states)
        ε = Series((0.0,1.0))

        # # analytical trace
        # Jac = batched_jacobian(func, AutoForwardDiff(), ϕ)
        # trj = map(tr,eachslice(Jac,dims=3))

        # hutchinson trace
        trj = @time trJ(rng,func,ϕ; ns=1)
        trj = mean(trj,dims=2)

        pt = Float64.(ϕ) .+ ε .* func(Float64.(ϕ));
        new = map(eachslice(pt,dims=ndims(pt))) do z z[:,:,1] end;
        cfg = map(cfgs) do z Float64.(z[:,:,1]) end;

        w = -(action.(new) .- ε .* source.(new)) .+ action.(cfg)
        w .+= ε .* trj
        w = exp.(w)
        w̃ = uw(w)

        y = reshape(w,1,:) .* dsum(pt,dims=Tuple(2:D+1))
        corri = uwcorr(y,ens_name) ./ w̃

        corri = getindex.(corri,2)
        corri ./= first(corri)
        uwerr.(corri)
##
    off =  1e-3

    p = plot(
        formatter=:latex,framestyle=:box,
        ylim=(off,2),
        xlim=(1,20),
        yscale=:log10,
        # tickfontsize=10,
        ylabel=L"C(x_0)",
        # xlabel=L"x_0",
        legend=:topright,
        size = (1000,700),
        guidefontsize=30, 
        tickfontsize=20, 
        legendfontsize=20,
        left_margin=10Plots.mm,
        bottom_margin=7Plots.mm,
        top_margin=7Plots.mm,
        # title=L"L=%$(first(vol))\times %$(last(vol)), \lambda=%$lam, \kappa=%$kap",
        titlefontsize=30
    )

    x = 1:first(vol)

    y,k = plottable(corr2; off=off)
    scatter!(p,x,y;k...,ms=10,c=jc.blue,msc=jc.blue,label="2pts function")
    # plot!(p,x,y;c=jc.blue,label="",alpha=0.4)
    
    y,k = plottable(corrw; off=off)
    scatter!(p,x.-0.2,y;k...,ms=7,m=:utriangle,c=jc.green,msc=jc.green,label="analytical")
    # plot!(p,x.-0.2,y;c=jc.green,label="",alpha=0.2)

    y,k = plottable(corri; off=off)
    scatter!(p,x.+0.2,y;k...,ms=7,m=:dtriangle,c=jc.red,msc=jc.red,label="effective prop.")
    # plot!(p,x.+0.2,y;c=jc.red,label="",alpha=0.2)


    display(p)
##
    p2 = plot(
        formatter=:latex,framestyle=:box,
        xlims=(1,20),
        # ylims=(0.3,0.38),
        # ylims=(0,1.),
        # tickfontsize=10,
        ylabel=L"m_\mathrm{eff}",
        xlabel=L"x_0",
        size = (1000,500),
        guidefontsize=30, 
        tickfontsize=20, 
        legendfontsize=20,
        left_margin=10Plots.mm,
        bottom_margin=7Plots.mm,
        top_margin=7Plots.mm,
        # title=L"L=%$(first(vol))\times %$(last(vol)), \lambda=%$lam, \kappa=%$kap",
        titlefontsize=30
    )
    
    x = 1:first(vol)

    ym = meff(corrw); uwerr.(ym)
    scatter!(p2,x.+0.1,value.(ym),yerrors=err.(ym),ms=10,c=jc.green,msc=jc.green,m=:utriangle,label="")

    ym = meff(corr2); uwerr.(ym)
    scatter!(p2,value.(ym),yerrors=err.(ym),m=10,c=jc.blue,msc=jc.blue,label="")

    ym = meff(corri); uwerr.(ym)
    scatter!(p2,x.-0.1,value.(ym),yerrors=err.(ym),ms=10,c=jc.red,msc=jc.red,m=:dtriangle,label="")
##
    plt = plot(p,p2,layout=(2,1),size=(900,1100))
    # savefig(plt,joinpath(PATH,"CORRELATOR_NN_$(ens_name).pdf"))
## ---------------------------------------------------
