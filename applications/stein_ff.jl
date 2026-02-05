using Pkg; Pkg.activate(".")
using SADflow, Revise
using LinearAlgebra, Random, StatsBase
using Lux, NNlib, MLUtils, Optimisers
using Zygote
using Plots, Pipe, Printf
using FormalSeries, Lattice



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
    f_true = analytical_transf(1/κ+8,vol)    
## ===============================================================

## ========================= Define model =========================
    model = FourierLatConv(20,5)

    conv() = 
    
    Chain(
        Conv((3,3),1=>2, tanh, pad=SamePad()),
        Conv((3,3),2=>1, tanh, pad=SamePad()),
    )
##
    seed  = 1994
    rng = Xoshiro(seed)
    ps, st = Lux.setup(rng, model)
## ==============================================================



## ======================= Define losses =======================
    function Stein(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T) where {T,N}
        func = StatefulLuxLayer(model,ps,st)
        f = func(ϕ)
        
        # Divergence: ∇⋅f = tr J_f 
        divf = div_FourierLatConv(ps,ϕ)
    
        # Score term f⋅∇log(r) = - f⋅∇S 
        f∇S = mean( Forceλϕ⁴(ϕ,λ,κ) .* f, dims=collect(1:N-1))[:]

        return f, divf .- f∇S
    end

    function score_mismatch(f::AbstractArray{T,N}, x₀::Int) where {T,N}
        return .- sum.(eachslice(selectdim(f,1,x₀),dims=N-1)) 
    end

    function phi4loss_KL2(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, nsources::Int, x₀::Int) where {T,N}
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
        σₒ² = selectdim(sum(ϕ,dims=2),1,x₀) |> var

        return mean(trJ²./2 .+ f⊥Hf./2 .+ f∇O) + σₒ²/2, st, (;)
    end

    function phi4loss_KL2stein(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, x₀::Int) where {T,N}
        # Stein operator
        f, Tᵣf = Stein(model,ps,st,ϕ,λ,κ) 
        
        # Score mismatch
        f∇O = score_mismatch(f,x₀)

        # Variance term
        σₒ² = sum.(eachslice(selectdim(ϕ,1,x₀),dims=ndims(ϕ)-1)) |> var

        return mean(( Tᵣf .^ 2)./2 .+ f∇O) + σₒ²/2, st, (;)
    end

    function phi4loss_OTnorm(model,ps,st,ϕ::AbstractArray{T,N}, λ::T, κ::T, x₀::Int; O_av=nothing) where {T,N}
        f,Tᵣf = Stein(model,ps,st,ϕ,λ,κ) # Stein operator

        # Control variates
        O = .- sum.(eachslice(selectdim(ϕ,1,x₀),dims=N-1))
        Op_av = isnothing(O_av) ? mean(O) : O_av
        Ō = O .- Op_av

        return norm((Tᵣf .- Ō),2), st, (;)
    end

    lossKL2(λ,κ;  nsources=20, x0=1         ) = (m,p,s,z) -> phi4loss_KL2(     m,p,s,z,λ,κ, nsources, x0)
    lossSt2(λ,κ;               x0=1         ) = (m,p,s,z) -> phi4loss_KL2stein(m,p,s,z,λ,κ, x0)
    lossOTS(λ,κ;               x0=1, O_av=0.) = (m,p,s,z) -> phi4loss_OTnorm(  m,p,s,z,λ,κ, x0; O_av=O_av)
## ==============================================================

## ========================== Training =============================
    NSAMPLES = size(ϕ,ndims(ϕ))

    # Metaparameters -----------
        x₀ = 1
        η = 0.001f0
        BSIZE = 50
    # --------------------------

    O_MC = .- selectdim(sum(ϕ,dims=2),1,x₀) |> mean

    LOSS1 = lossKL2(λ,κ; nsources=1, x0=x₀)
    LOSS2 = lossSt2(λ,κ;             x0=x₀)
    LOSS3 = lossOTS(λ,κ;             x0=x₀, O_av=O_MC)

    ts = Training.TrainState(model, ps, st, Optimisers.Adam(η));
##
    metrics = (kl2=[],stein=[],ot=[],delta=[])
##
    zin = ϕ[:,:,:,1:BSIZE]
    LOSS2(model,ps,st,zin)

    _, loss, metr, ts = @time Training.single_train_step!( 
        AutoZygote(),  
        LOSS2, zin, ts;
    );

##    
    for epoch in 1:100
        stime = time()
        for i in 1:NSAMPLES÷BSIZE
            i_samples = sample(rng,1:NSAMPLES,BSIZE,ordered=true)
            ϕ_sample = ϕ[:,:,:,i_samples]

            _, loss, mtr, ts = Training.single_train_step!( 
                AutoZygote(),  
                LOSS1, ϕ_sample, ts;
            );
        end
        ttime = time()-stime

        l1,_,s1 = LOSS1(model,ts.parameters, ts.states, ϕ) # old
        l2,_,s2 = LOSS2(model,ts.parameters, ts.states, ϕ) # stein
        l3,_,s3 = LOSS3(model,ts.parameters, ts.states, ϕ) # ot

        f,_ = Lux.apply(model,ϕ,ts.parameters, ts.states)
        δ = norm(f_true .- f)

        # Logs
        @printf("epoch=%i [%.3f s]: %.7f %.7f %.7f [%.7f]\n",epoch, ttime, l1, l2, l3, δ)
        push!(metrics.kl2,l1)
        push!(metrics.stein, l2)
        push!(metrics.ot, l3)
        push!(metrics.delta, δ)
    end
##

    plt1 = plot(metrics.kl2  , label="KL₂"               ,yscale=:log10)
    plt2 = plot(metrics.stein, label="Stein"             ,yscale=:log10)
    plt3 = plot(metrics.ot   , label="||Tᵣf-(O-⟨O⟩)||²)" ,ylim=(0,100))
    plt4 = plot(metrics.delta, label="||f-f_true||²"     ,yscale=:log10)

    title = "fourier features"

    p = plot(
        plt1,plt2,plt3,plt4,
        layout=(4,1),
        link=:x,
        size=(400,500),
        plot_title=title,
        plot_titlefontsize = 12
    )

    display(p)
    # savefig(p,"./applications/plots/$title.pdf")
##
    fϕ,_ = model(ϕ,ts.parameters,ts.states)

    plt1 = heatmap(ϕ[:,:,1,2])
    plt2 = heatmap(fϕ[:,:,1,2])
    plt3 = heatmap(f_true)
    plot(plt1,plt2,plt3,layout=(1,3),size=(1300,500))
## ==============================================================

A = ϕ[:,:,1,2]
B = fϕ[:,:,1,2]
C = f_true

# global color limits
cl =  extrema(B)

plt1 = heatmap(A; clims=cl, colorbar=true)
plt2 = heatmap(B; clims=cl, colorbar=false)
plt3 = heatmap(C; clims=cl, colorbar=false)

plot(plt1, plt2, plt3, layout=(1,3), size=(1000,500))

