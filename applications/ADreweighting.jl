using Pkg; Pkg.activate(".")
using SADflow, Revise
using LinearAlgebra, Random, StatsBase
using Lux, NNlib, MLUtils, Optimisers
using Zygote
using Plots, Pipe, Printf
using FormalSeries, Lattice
using ADerrors



## ============== Read configuration from ensemble ==============
    folder = "/Users/pietro/code/software/SADflow/.local_sketches/2d_l0.0_k0.12461_L_8_8"
    vol = (8,8)
    (κ,λ,α) = (0.12461f0,0.0f0,1.f0)
    # folder = "/Users/pietro/code/software/SADflow/.local_sketches/2d_l0.0_k0.12461_L_128_128"
    # vol = (128,128)
    # (κ,λ,α) = (0.12461f0,0.0f0,1.f0)

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
    f_true = analytical_transf((1-2λ)/κ-2*length(vol),vol)    
## ===============================================================
    using FFTW

    L = 128
    m² = 1/κ-2*length(vol)

    p̂ = collect(0:L-1) .* 2π/L


    ft = zeros(Float64,L,L)
    ft[:,1] .= .5 ./ (4 .*sin.(p̂./2).^2 .+ m²)
    plot(ft[:,1])

    heatmap(ft)

    ftr = ifft(ft)

    p1 = heatmap(real.(ftr))
    p2 = heatmap(imag.(ftr))
    plot(p1,p2,size=(500,250))

    pp = real.(ftr[:,1])
    pp[pp .< 1e-15] .=  NaN
    scatter(pp,yscale=:log10)
    # plot(abs.(real.(ftr[1,:])),yscale=:log10)

## ================================================================
    vjv = (rng, func, z)-> begin
        η = rand(rng,[-1,1],size(z)...)
        Jη = jacobian_vector_product(func, AutoForwardDiff(), z, η)
        @assert ndims(η)==ndims(Jη)
        eachslice(η .* Jη,dims=ndims(η)) .|> sum
    end

    trJ(rng, func, z; ns=1) = 
        [vjv(rng,func,z) for _ in 1:ns] |> stack
    trJ(func, z; ns=1) = 
        trJ(Random.default_rng(1), func,z; ns=ns)

## ================================================================

    der(x::Series) = getindex(x,2) 


    action(ϕ) = -2κ .* ϕ .* neighbor_sum(ϕ) .+ ϕ.^2 .+ λ .* (ϕ.^2 .- 1).^2    
    
    SADflow.Actionλϕ⁴(ϕ::AbstractArray{Series{T,D},N},λ::T,κ::T) where {D,T,N} = (
        -2κ .* ϕ .* neighbor_sum(ϕ) .+ ϕ.^2 .+ λ .* (ϕ.^2 .- T(1)).^2
    ) |> x->sum(x; dims=1:N-1)

    model = Chain(
        FlattenLayer(),
        Dense(prod(vol)=>prod(vol)),
        Dense(prod(vol)=>prod(vol)),
        ReshapeLayer((vol...,1))
    )
    ps,st = Lux.setup(Xoshiro(1994),model)

    FF = f_true .+ randn_like(f_true)./10000
    f_train = fill(FF[:,:,:],10000) |> stack
##
    train_state = Training.TrainState(model, ps, st, Adam(0.01f0))

    for step in 1:1000
        gs, loss, stats, train_state = Lux.Training.single_train_step!(
            AutoZygote(),
            MSELoss(),
            (ϕ,f_train),
            train_state
        )

        @show step, loss
    end
##

    func = StatefulLuxLayer(model,train_state.parameters,train_state.states)

## ==================================================================

    ens = "this"

    # 2pt function
        corr_2 = sum(ϕ[1:1,:,:,:] .* ϕ,dims=2)[:,1,1,:]
        corr_2 = map(x-> uwreal(Float64.(x),ens), eachrow(corr_2))
        corr_2 = corr_2 
        uwerr.(corr_2)

    # AD
        S̃ = Actionλϕ⁴(ϕ,λ,κ)[:] .- Series((0.f0,1.f0)) .* sum(ϕ[1,:,1,:],dims=1)[:]
        ΔS = S̃ .- Actionλϕ⁴(ϕ,λ,κ)[:]
        w̃ = exp.(-ΔS)

        Ew = uwreal(der.(w̃) .|> Float64,ens)

        w̃ϕ = sum(reshape(w̃,1,1,1,:) .* ϕ,dims=2)[:,1,1,:]
        corr_AD = map(x-> uwreal(Float64.(x),ens), eachrow(der.(w̃ϕ)))
        corr_AD = [Ew * c for c in corr_AD]
        uwerr.(corr_AD)



    # Exact reweigthing
        ϕ̃ = ϕ .+ Series((0.f0,1.f0)) .* f_true
        corr_ex = sum(ϕ̃,dims=2)[:,1,1,:]
        corr_ex = map(x-> uwreal(Float64.(x),ens), eachrow(der.(corr_ex)))
        uwerr.(corr_ex)

    # # Improved reweighting
        ϕ̃ = ϕ .+ Series((0.f0,1.f0)) .* func(ϕ)

        Jac = batched_jacobian(func, AutoForwardDiff(), ϕ)
        trj = tr.(eachslice(Jac,dims=3))
        # logdetJ = [Series((0.f0,d1)) for d1 in mean(trJ(func,ϕ),dims=2)[:]]
        logdetJ = [Series((0.f0,d1)) for d1 in trj]

        S̃ = Actionλϕ⁴(ϕ̃,λ,κ)[:] .- Series((0.f0,1.f0)) .* sum(ϕ̃[1,:,1,:],dims=1)[:]
        ΔS = S̃ .- Actionλϕ⁴(ϕ,λ,κ)[:]
        w̃ = exp.( -ΔS .+ logdetJ )

        Ew = uwreal(der.(w̃) .|> Float64,ens)

        w̃ϕ = sum(reshape(w̃,1,1,1,:) .* ϕ,dims=2)[:,1,1,:]
        corr_irw = map(x-> uwreal(Float64.(x),ens), eachrow(der.(w̃ϕ)))
        corr_irw = [Ew * c for c in corr_irw] 
        uwerr.(corr_irw)


##
    effm(corr) = log.(circshift(corr,-1)./corr)
##
    plt = plot(yscale=:log10,ylim=(1e-5,100))

        scatter!(plt,value.(corr_2),yerror=ADerrors.err.(corr_2),label="2pts")
        # scatter!(plt,value.(corr_AD),yerror=ADerrors.err.(corr_AD),label="AD")
        scatter!(plt,value.(corr_ex),yerror=ADerrors.err.(corr_ex),label="exact rew.")
        scatter!(plt,value.(corr_irw),yerror=ADerrors.err.(corr_irw),label="improved rw.")
    
    
    display(plt)
##    