using JLD2, Pipe, StatsBase, Random, Plots, LinearAlgebra, Revise, FFTW, Statistics, DataFrames
using Lattice, SADflow
using ADerrors, FormalSeries
using ForwardDiff, Lux

## ================================ Utilities ================================
    _action(λ,κ) = ϕ ->
        ϕ.^2 .+ λ .* (ϕ.^2 .- 1).^2 .+
        -2κ .* ϕ .* (
            circshift(ϕ,(-1,0)) .+ circshift(ϕ,(0,-1))
        ) |> sum

    _grad_S(λ,κ) = ϕ -> 
        2ϕ .+ 4λ .* (ϕ.^2 .- 1) .* ϕ .+ 
        -2κ .* (
            circshift(ϕ,(-1,0)) .+ circshift(ϕ,(1,0)) .+
            circshift(ϕ,(0,-1)) .+ circshift(ϕ,(0,1)) 
        )

    _Hvp(λ,κ) = (ϕ,v) -> 
        (2 .- 4λ .+ 12λ .* ϕ.^2) .* v .+ 
        -2κ .* (
            circshift(v,(-1,0)) .+ circshift(v,(1,0)) .+
            circshift(v,(0,-1)) .+ circshift(v,(0,1)) 
        )
    
    deltalike(z; xstar=1) = begin
        δ = zeros(eltype(z),size(z)...)
        selectdim(δ,1,xstar) .= one(eltype(z))
        return δ
    end
    
    obs(z; x₀=1) = dsum(z,dims=2)[x₀] 
## ==========================================================================

## =================== Load pre-generated configurations ===================
    # Paths
    FOLDER   = "/Users/pietro/code/software/SADflow/.local_sketches/"
    ENS_NAME = "2d_l0.05_k0.12461_L_32_8"

    data      = load(joinpath(FOLDER, "$ENS_NAME.jld2"))
    vol, λ, κ = data["vol"], data["λ"], data["κ"]
    ϕ         = data["ϕ"][:,:,1,:]

    λ,κ = Float64.((λ,κ))
    cfgs = eachslice(ϕ, dims=ndims(ϕ))
##
    const action = _action(λ,κ)
    const grad_S = _grad_S(λ,κ)
    const Hvp    = _Hvp(λ,κ)

    const O₀ = uwreal(Obs(ϕ)[:],ENS_NAME); uwerr(O₀)
    const ΔO₀ = mchist(O₀,ENS_NAME)

    const f0 = free_propagator(vol,κ,λ)
    const δ₀ = deltalike(first(cfgs))

    Δf = fout -> begin
        c = sumnorm(fout)
        f = sumnorm(f0)
        abs.(c .- f) ./ f
    end
## ==========================================================================


## ========================= Langevin methods ================================
    LangevinStep(Zₜ, ∇S, Δt, ηₜ) = Zₜ .- Δt .* ∇S .+ sqrt(2Δt)*ηₜ
    AdjointLangevinStep(Aₛ, Hₛv, Δt, d₀)   = Aₛ .+ Δt .* (-Hₛv .+ d₀)

    function FeynmanKac(z₀,η,δt)
        itr = eachslice(η,dims=ndims(η))
        (Z_T,traj) = foldl(itr; init=(z₀,())) do (Zₜ,traj), ηₜ
            Zₜ = LangevinStep(Zₜ,grad_S(Zₜ),δt,ηₜ)
            (Zₜ,(traj...,Zₜ))
        end

        A₀ = zeros(eltype(Z_T),size(Z_T)...)
        int = foldl(reverse(traj); init=A₀) do Aₛ, Zₜ
            HₛA = Hvp(Zₜ,Aₛ)
            Aₛ = AdjointLangevinStep(Aₛ,HₛA,δt,δ₀)
        end

        return int
    end
## ==========================================================================

    const SEED = 1994
    rng = Xoshiro(SEED)


## Check variance of 
    Tmax,δt = 1000, 0.1
    noise = randn(rng,vol...,Tmax)
    fk = z -> FeynmanKac(z,noise,δt)
    
    fout = cfgs[1:100] .|> fk |> stack
    heatmap(std(fout,dims=3)[:,:,1],scale=:log10)


##  --- Check ∇⋅f == ∇S⋅f - ΔO₀ --------------------------------------------
    IDX = 1
    Nsources = 200

    z₀ = cfgs[IDX]
    ∇S = grad_S(z₀)

    df = DataFrame(
        Tmax=[], ns=[], divf=[], trjf=[]
    )

    for Tmax in [50,100,200,300,500,1000,2000]
        noise = randn(rng,vol...,Tmax,Nsources)
        itr = eachslice(noise,dims=ndims(noise))
        for (i,η) in enumerate(itr)
            FK = z -> FeynmanKac(z,η,δt)
            fout = FK(z₀)

            divf = ∇S⋅fout - ΔO₀[IDX]
            trjf = tr(ForwardDiff.jacobian(FK,z₀))
            
            divf,trjf

            push!(df,(Tmax=Tmax, ns=i, divf=divf, trjf=trjf))
            @show Tmax,i, divf, trjf
        end
    end

    # divf = getindex.(X,1)
    # trjf = getindex.(X,2)

    # plot(cumsum(divf) ./ (1:Nsources), label="Stein Poisson")
    # plot!(cumsum(trjf) ./ (1:Nsources), label="AD")

    using Colors
    c = palette(:lipari,7)

    global i = 1
    p = plot()
    for d in groupby(df,:Tmax)
        tmax = d.Tmax|>unique|>only

        c1 = cumsum(d.divf) ./ (1:maximum(d.ns))
        c2 = cumsum(d.trjf) ./ (1:maximum(d.ns))

        plot!(p,abs.(c1.-c2), color=c[i],label=tmax)

        i+=1
    end
    display(p)

    dif = [abs((mean(d.divf)-mean(d.trjf))/mean(d.divf)) for d in groupby(df,:Tmax,sort=true)]
    err = [abs((mean(d.divf)-mean(d.trjf))/mean(d.divf)) for d in groupby(df,:Tmax,sort=true)]
    scatter(unique(df.Tmax),dif)
## Check trJ via finite differences
    IDX = 363
    Tmax,δt = 1000, 0.1
    noise = randn(rng,vol...,Tmax)
    fk = z -> FeynmanKac(z,noise,δt)

    z₀ = cfgs[IDX]

    trjf = tr(ForwardDiff.jacobian(fk,z₀))

    # cross-check trJ with finite differences
    ε = 1e-5
    trJ_fd = sum(eachindex(z₀)) do i
        z₊, z₋ = copy(z₀), copy(z₀)
        z₊[i] += ε; z₋[i] -= ε
        (fk(z₊)[i] - fk(z₋)[i]) / (2ε)
    end

    trjf = tr(ForwardDiff.jacobian(fk,z₀))

    @show trJ_fd, trjf
## ---------------------------------------------------------------------------
## Check propagator
    rng = Xoshiro(1994)

    Tmax,δt = 1000, 0.1
    noise = randn(rng,vol...,Tmax)
    fk = z -> FeynmanKac(z,noise,δt)

    z₀ = cfgs[4]
    f1 = fk(z₀)

    p1 = plot(sumnorm(f0),yscale=:log10,label="TRW")
    scatter!(p1, sumnorm(f1),yscale=:log10,label="Langevin")

    p2 = scatter(Δf(f1),label="rel diff",yscale=:log10,ylims=(1e-12,Inf))

    plot(p1,p2,layout=(2,1),link=:x)

## Check correlator
    ε = Series((0.,1.))
    z = stack(cfgs[1:100])

    # 2pt function
        corr2 = dsum(z[1:1,:,:] .* z,dims=2)
        corr2 = map(eachrow(corr2)) do x uwreal(x[:],"ciao") end 
        
        corr2 ./= first(corr2)
        uwerr.(corr2)

    # Exact reweighting
        c2 = dsum(z .+ ε .* f0,dims=2) 
        corrw = map(eachrow(getindex.(c2,2))) do r uwreal(r[:],"TRW (free th.)") end 
        corrw ./= first(corrw)
        uwerr.(corrw)

    # Improved reweigthing
        Tmax,δt = 1000, 0.1
        X = map(eachslice(z,dims=ndims(z))) do z₀
            noise = randn(rng,vol...,Tmax)
            fk = z -> FeynmanKac(z,noise,δt)

            @show mean(noise)
            
            fout = fk(z₀)
            trjf = tr(ForwardDiff.jacobian(fk,z₀))

            (fout,trjf)
        end
##
        fout = getindex.(X,1) |> stack
        trjf = getindex.(X,2) 

        w = compute_weight(z,fout,action,obs,trjf)
        corri = reweight_corr(w,z .+ ε.*fout,"ciao")
        corri ./= first(corri)

        uwerr.(corri)    
##
    p = plot(ylim=(1e-13,Inf),yscale=:log10)
    scatter!(p,value.(corr2),yerror=ADerrors.err.(corr2))
    plot!(p,value.(corrw))
    scatter!(p,value.(corri))

## PLOT
    effm(corr; half=true) = begin
        m = circshift(corr,-1) ./ corr
        m = -log.(ifelse.(value.(m).>0,m,missing))
        m = m[1: (half ? length(corr)÷2 :  end)]
    end 
    ADerrors.uwerr(::Missing) = 0
    ADerrors.value(::Missing) = missing
    ADerrors.err(::Missing)   = 0.


    p = plot(ylim=(1e-13,Inf),yscale=:log10)
    scatter!(p,value.(corr2),yerror=ADerrors.err.(corr2))
    plot!(p,value.(corrw))
    scatter!(p,value.(corri))
##
    eff2 = effm(corr2, half=false); uwerr.(eff2)
    effi = effm(corri, half=false); uwerr.(effi)
    effw = effm(corrw, half=false); uwerr.(effw)
    
    m = plot(xlim=(0,first(vol)/2),ylims=(1.5,2))
    # ymax = filter(!ismissing,eff2) .|> value |> maximum
    # m = plot(xlim=(0,first(vol)/2),ylim=(-ymax,ymax).*1.1)
    # scatter!(m, value.(eff2),yerror=ADerrors.err.(eff2))
    scatter!(m, value.(effi),yerror=ADerrors.err.(effi))
    scatter!(m, value.(effw),yerror=ADerrors.err.(effw))
    scatter!(m, value.(eff2),yerror=ADerrors.err.(eff2))

    plot(p,m,layout=(2,1),link=:x)
