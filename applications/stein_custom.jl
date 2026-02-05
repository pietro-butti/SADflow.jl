using Pkg; Pkg.activate(".")
using SADflow, Revise
using LinearAlgebra, Random, StatsBase
using Lux, NNlib, MLUtils, Optimisers
using Zygote
using Plots, Pipe, Printf, DataFrames, Colors
using FormalSeries, Lattice
using ADerrors


## ============== Read configuration from ensemble ==============
    folder = "/Users/pietro/code/software/SADflow/.local_sketches/2d_l0.0_k0.12461_L_8_8"
    vol = (8,8)
    (κ,λ,α) = (0.12461f0,0.0f0,1.f0)

    # Infer configuration indexes -----------------------------
    ens_name = split(folder,"/") |> last
    conflist = filter(x->endswith(x,".jld2"),readdir(folder))
    str_idx = map(x->split(x,".")[end-1][4:end], conflist)
    confidx = parse.(Int,str_idx)

    space = Grid{length(vol)}(vol,Tuple([8 for _ in vol]))
    ws    = Phi4_workspace(space)
    
    # Setup readers -------------------------------------------
    suffix = i -> ".cfg$(lpad(i,5,'0')).jld2"
    pics = Array{Float32}(undef, space.iL...,1,0)
    for icfg in confidx
        Lattice.Fields.read!(ws._phi, "$folder/$ens_name$(suffix(icfg))")
        pics = cat(pics,conf_to_pic(ws._phi,space),dims=ndims(pics))
    end

    ϕ = Float32.(pics)
## ===============================================================
 
## =========================== Utils =============================
    function analytical_transf(m̂²::T,vol) where T
        L₀ = first(vol)
        V  = prod(vol)

        p(n) = 2π/L₀*n
        p̂(n) = 2*sin(p(n)/2)

        f = x₀ -> [
            exp(-im * p(n) * x₀)/( (2*p̂(n))^2 + m̂²)/2 / V
            for n in (1:L₀).-1
        ] |> sum

        f_t = f.((1:L₀).-1)
        f_t = ifelse.( norm.(f_t).>1e-10, f_t, 0 )
        f_t = ifelse.( imag.(f_t).>1e-10, f_t, real.(f_t))

        return f_t .* ones(T,vol...)
    end
    f_true = analytical_transf(1/κ-2*length(vol),vol)  

    function Stein_violation(model,ps,st,ϕ)
        func = StatefulLuxLayer(model,ps,st)
        f = func(ϕ)

        aux = [trace_J2(func,ϕ) for _ in 1:1] |> stack
        trJ2 = mean(aux,dims=2)[:]

        f,Trf = Stein(model,ps,st,ϕ,λ,κ,1)
        fhf = fHf(ϕ,f,λ,κ)[:]

        Tᵣf  = uwreal(Float64.(Trf),"this");  
        trJ² = uwreal(Float64.(trJ2),"this"); 
        Tᵣf² = uwreal(Float64.(Trf).^2,"this");
        _fhf = uwreal(Float64.(fhf),"this"); 

        δ = Tᵣf; uwerr(δ)
        Δ = - trJ² + Tᵣf² - _fhf; uwerr(Δ)

        return (δ,Δ), (trJ2,Trf.^2,fhf)
    end
## ===============================================================

## ====================== Loss functions ======================
    function Stein(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int) where {T,N}
        func = StatefulLuxLayer(model,ps,st)
        f = func(ϕ)
        
        # Divergence: ∇⋅f = tr J_f 
        trJ = trace_J(func,ϕ,nsources)
        divf = mean(trJ,dims=2)[:]

        # Score term f⋅∇log(r) = - f⋅∇S
        f∇S = sum(Forceλϕ⁴(ϕ,λ,κ) .* f, dims=1:(N-1))[:]

        return f, divf .- f∇S
    end

    function score_mismatch(f::AbstractArray{T,N}, x₀::Int) where {T,N}
        return sum.(eachslice(selectdim(f,1,x₀),dims=N-1)) 
    end
##
    function lSt2(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int, x₀::Int; O_av=nothing) where {T,N}
        # Stein operator
        f, Tr_f = Stein(model,ps,st,ϕ,λ,κ,nsources) 
        
        # Score mismatch
        f∇O = score_mismatch(f,x₀)

        # Variance term
        σₒ² = sum.(eachslice(selectdim(ϕ,1,x₀),dims=ndims(ϕ)-1)) |> var

        return σₒ² + mean(Tr_f.^2 .+ 2f∇O), st, (;)
    end
    lossSt2(λ,κ;  nsources=20, x0=1, O_av=0.f0) = (m,p,s,z) -> lSt2(m,p,s,z,λ,κ, nsources, x0; O_av=O_av)

    function KL2(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int, x₀::Int) where {T,N}
        func = StatefulLuxLayer(model,ps,st)
        f = func(ϕ)
        
        # Hessian term
        f⊥Hf = fHf(ϕ,f,λ,κ)[:] 

        # Stochastic Jacobian trace
        trJ² = mean(
            [trace_J2(func,ϕ) for _ in 1:nsources] |> stack,
            dims=2
        )[:]

        # Score mismatch
        f∇O = score_mismatch(f,x₀)

        # Variance term
        σₒ² = sum.(eachslice(selectdim(ϕ,1,x₀),dims=ndims(ϕ)-1)) |> var

        return  σₒ² + mean(trJ² .+ f⊥Hf .+ 2f∇O), st, (;)
    end
    lossKL2(λ,κ;  nsources=20, x0=1) = (m,p,s,z) -> KL2(m,p,s,z,λ,κ, nsources, x0)

    function lOT(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int, x₀::Int; O_av=nothing) where {T,N}
        f,Tr_f = Stein(model,ps,st,ϕ,λ,κ,nsources) # Stein operator

        # Control variates
        O = .- sum.(eachslice(selectdim(ϕ,1,x₀),dims=N-1))
        Op_av = isnothing(O_av) ? mean(O) : O_av
        Ō = O .- Op_av

        return norm(Tr_f .- Ō,2), st, (;)
    end
    lossOT2(λ,κ;  nsources=20, x0=1, O_av=0.f0) = (m,p,s,z) -> lOT(m,p,s,z,λ,κ, nsources, x0; O_av=O_av)
## ==============================================================



## ========================= Define model =========================
## Convolutions
    # --------- my original one
    model = Chain(
        pad,Conv((3,3),1=>2, tanh; use_bias=false),
        pad,Conv((3,3),2=>1, tanh; use_bias=false),
    )

    # # --------- louis one
    # model = Chain(
    #     Conv((3,3),1=>2, tanh, pad=SamePad()),
    #     Conv((3,3),2=>1, tanh, pad=SamePad()),
    # )


## Normalising flow
    net() = Chain(
        Conv((3,3),1=>2, tanh, pad=SamePad()),
        Conv((3,3),2=>2, tanh, pad=SamePad()),
        x->biject(x; dim=3)
    )
    model = Chain(
        AffineEOCoupling(net(),vol,true),
        AffineEOCoupling(net(),vol,false),
    )
## Fourier features
    pad = WrappedFunction(x->pad_circular(x,(2,2,2,2)))
    net() = Chain(
        pad,Conv((5,5),1=>1, identity; use_bias=false),
    )

    model = BranchLayer(
        [FourierFeature(net(),glorot_normal) for _ in 1:10]...;
        fusion = (xs...) -> reduce(+,xs)
    )
## MLP

    net(vol) = @compact(
        layer = Dense(prod(vol)=>prod(vol))
    ) do x_in
        x = MLUtils.flatten(x_in)
        x = layer(x)
        x = exp.(x)
        @return reshape(x,size(x_in))
    end

    model = net((8,8))

##
    seed  = 1994
    rng = Xoshiro(seed)
    ps, st = Lux.setup(rng, model)
## ==============================================================

## ========================== Training =============================
    # Metaparameters -----------
        x₀       = 1
        NSRC     = 1
        η        = 0.001f0
        BSIZE    = 50
        NSAMPLES = 1000
    # --------------------------

    O_MC = sum.(eachslice(selectdim(ϕ,1,x₀),dims=ndims(ϕ)-1)) |> mean

    LOSS1 = lossSt2(λ,κ; nsources=NSRC, x0=x₀, O_av=O_MC)
    LOSS2 = lossKL2(λ,κ; nsources=NSRC, x0=x₀)
    LOSS3 = lossOT2(λ,κ; nsources=NSRC, x0=x₀, O_av=O_MC)

    ts = Training.TrainState(model, ps, st, Optimisers.Adam(η));
    ϕtrain = ϕ[:,:,:,1:NSAMPLES]
    ϕtest  = ϕ[:,:,:,1001:2000] 
##
    metrics = DataFrame(
        stein = [], stein_test = [],
        kl2   = [], kl2_test   = [],
        norm  = [], norm_test  = [],
        delta = [],
        delta_stein=[], delta_stein2=[]
    )

    _ftrue = analytical_transf(1/κ+8,vol)
    function compute_metrics(model,ps,st,z; violation=false)
        l1,_,_ = LOSS1(model,ps,st,z)
        l2,_,_ = LOSS2(model,ps,st,z)
        l3,_,_ = LOSS3(model,ps,st,z)

        if violation
            (δ,Δ), _ = Stein_violation(model,ts.parameters,ts.states,ϕ)
        else
            (δ,Δ) = (NaN,NaN)
        end

        return (l1,l2,l3,(δ,Δ))
    end

##
    zin = ϕ[:,:,:,1:BSIZE]
    LOSS1(model,ps,st,zin)

    _, loss, metr, ts = @time Training.single_train_step!( 
        AutoZygote(),  
        LOSS2, zin, ts;
    );

##    
    for epoch in 1:1000
        stime = time()
        for i in 1:NSAMPLES÷BSIZE
            i_samples = sample(rng,1:NSAMPLES,BSIZE,ordered=true)
            ϕ_sample = ϕtrain[:,:,:,i_samples]

            _, loss, mtr, ts = Training.single_train_step!( 
                AutoZygote(),  
                LOSS2, ϕ_sample, ts;
            );
        end
        ttime = time()-stime


        # Training logs ----------
            (_ps,_st) = ts.parameters, Lux.testmode(ts.states)

            l1,l2,l3,(δ,Δ) = compute_metrics(model, _ps,_st, ϕtrain; violation=epoch%10==0)
            t1,t2,t3, _    = compute_metrics(model, _ps,_st, ϕtest)

            fout,_ = model(ϕtrain,_ps,_st)
            D = norm.(eachslice(1 .- _ftrue ./ fout,dims=4)) |> mean

            push!(metrics,(l1,t1,l2,t2,l3,t3,D,δ,Δ))
            
            @printf("epoch=%i [%.3f s]: %.7f %.7f %.7f %.7f\n",epoch, ttime, l1, l2, l3, D)
            
    end



##
    import ADerrors.value, ADerrors.err
    value(x::T) where T<:AbstractFloat = x
    err(x::T) where T<:AbstractFloat = T(0)

    c = Colors.JULIA_LOGO_COLORS

    plt1 = plot(metrics.stein,yscale=:log10,label="Stein",c=c.blue)
    plot!(plt1,metrics.stein_test,yscale=:log10,label="",c=c.blue,ls=:dash)
    
    plot!(plt1,metrics.kl2,label="KL₂",c=c.red)
    plot!(plt1,metrics.kl2_test,label="",c=c.red,ls=:dash)

    plt2 = plot(metrics.norm, yscale=:log10, label="OT norm", c=c.green)
    
    plt3 = plot(metrics.delta, label="|f-f_true|/f_true", c=c.purple)
    
    plt4 = scatter(value.(metrics.delta_stein2),yerr=err.(metrics.delta_stein2),yscale=:log10,label="Boundary term",c=c.purple)

    plot(plt1,plt2,plt3,plt4,layout=(4,1),link=:x,size=(500,500))
##
    fϕ,_ = model(ϕ,ts.parameters,ts.states)

    plt1 = heatmap(ϕ[:,:,1,2])
    plt2 = heatmap(fϕ[:,:,1,2])
    plt3 = heatmap(f_true)

    plot(plt1,plt2,plt3,layout=(1,3),size=(1000,300))

##
    A = ϕ[:,:,1,2]
    B = fϕ[:,:,1,2]
    C = f_true

    # global color limits
    cl =  extrema(B)

    plt1 = heatmap(A; clims=cl, colorbar=true)
    plt2 = heatmap(B; clims=cl, colorbar=false)
    plt3 = heatmap(C; clims=cl, colorbar=false)

    plot(plt1, plt2, plt3, layout=(1,3), size=(1000,500))
##
## ==============================================================


