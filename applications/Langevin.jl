using Pkg; Pkg.activate("."); using SADflow
using JLD2, Plots, Random, Pipe, StatsBase, LinearAlgebra
using ADerrors, FormalSeries, ForwardDiff, Lux
using Plots, LaTeXStrings

jc = Colors.JULIA_LOGO_COLORS
import ADerrors.err
## ==========================================================
@show "READING DATA"
    folder   = "/Users/pietro/code/software/SADflow/.local_sketches/"
    ens_name = "2d_l0.1_k0.12461_L_8_8" 
    
    data    = load(joinpath(folder,"$ens_name.jld2"))
    vol,λ,κ = data["vol"],data["λ"],data["κ"]

    λ = Float64(λ)
    κ = Float64(κ)

    ϕ = data["ϕ"][:,:,1,:]
    ϕ = ϕ[:,:,sort(rand(last(axes(ϕ)),100))]
    
    D = length(vol)
    cfgs = eachslice(ϕ,dims=ndims(ϕ))
##
    LangevinStep(Zₜ, ∇S, Δt, ηₜ) = Zₜ .- Δt .* ∇S .+ sqrt(2Δt)*ηₜ
    AdjointLangevinStep(Aₛ, Hₛv, Δt, d₀)   = Aₛ .+ Δt .* (-Hₛv .+ d₀)

    function FeynmanKac(z₀,η,δt; grad_S=nothing, Hvp=nothing, δ₀=nothing)
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

    deltalike(z; xstar=1) = begin
        δ = zeros(eltype(z),size(z)...)
        selectdim(δ,1,xstar) .= one(eltype(z))
        δ
    end

    action = _action(λ,κ)
    grad_S = _grad_S(λ,κ)
    Hvp    = _Hvp(λ,κ)

    δ₀ = deltalike(first(cfgs))
    FK(args...) = FeynmanKac(args...; grad_S=grad_S,Hvp=Hvp,δ₀=δ₀)

    VJV = (rng, func, z)-> begin
        η = rand(rng,[-1,1],size(z)...) .|> eltype(z)
        Jη = jacobian_vector_product(func, AutoForwardDiff(), z, η)
        η⋅Jη
    end

    TRJ(rng, func, z; ns=1) = [VJV(rng,func,z) for _ in 1:ns]
    TRJ(func, z; ns=1) = TRJ(Random.default_rng(), func,z; ns=ns)
## ===========================================================

    seed = 1994
    Tmax = 500
    δt   = 0.1

    rng = Xoshiro(seed)

    cnt = 0
    divf1, divf2 = 0, 0

    Oav = mean(source.(cfgs))
    
    out = map(cfgs) do z₀ 
        noise = randn(rng,vol...,Tmax)
        fk = z->FK(z,noise,δt) 
        f = fk(z₀)

        divf1 = tr(ForwardDiff.jacobian(fk,z₀))
        divf2 = TRJ(rng, fk, z₀; ns=10)
        
        f∇S = grad_S(z₀) ⋅ f

        global cnt += 1
        @show  cnt, mean(divf2) - f∇S + source(z₀)

        (f,divf1,divf2)
    end

    # trjs = stack(getindex.(out,3))
    trjs = stack(getindex.(out,2))
    fout = getindex.(out,1)


    # trjs = grad_S.(cfgs) .⋅ fout .- source.(cfgs)




## --- Variance reduction ----------------------------------------------------
    ε = Series((0.,1.))

    Oav = mean(source.(cfgs))

    m²(κ,λ,vol) = (1-2λ)/2κ - 2*length(vol)


    # 2pt function
        x = sumvol(ϕ,dims=Tuple(2:D))
        corr2 = uwcorr(x[1:1,:] .* x,"scemo")
        corr2 ./= first(corr2)
        uwerr.(corr2)

    # Improved reweighting (max uncoupled noise)
        ztld = map(zip(cfgs,fout)) do (z,f) @. z+ε*f end

        w = -(action.(ztld) .- ε .* source.(ztld)) .+ action.(cfgs)
        w .+= ε .* dmean(trjs,dims=1)
        w = exp.(w)
        w̃ = uwreal(w,ens_name)

        y = reshape(w,1,:) .* dsum(stack(ztld),dims=Tuple(2:D))
        corri = uwcorr(y,ens_name) ./ w̃

        corri = getindex.(corri,2)
        corri ./= first(corri)
        uwerr.(corri)
    
    # Exact reweighting




    

##
    off =  1e-3

    lam = round(λ,digits=2)
    kap = round(κ,digits=3)

    p = plot(
        formatter=:latex,framestyle=:box,
        ylim=(off,2),
        # xlim=(1,20),
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
        title=L"L=%$(first(vol))\times %$(last(vol)), \lambda=%$lam, \kappa=%$kap",
        titlefontsize=30
    )

    x = 1:first(vol)

    y,k = plottable(corr2; off=off)
    scatter!(p,x,y;k...,ms=7,c=jc.blue,msc=jc.blue,label="2pts")
    
    # y,k = plottable(corrw; off=off)
    # scatter!(p,x.-0.2,y;k...,ms=7,m=:utriangle,c=jc.green,msc=jc.green,label="analytical")

    y,k = plottable(corri; off=off)
    scatter!(p,x.+0.2,y;k...,ms=7,m=:dtriangle,c=jc.red,msc=jc.red,label="FK")


    display(p)
    PATH = "/Users/pietro/Documents/Physics/Talks_and_presentations/2025-2026/QTC2026/"

    # savefig(p,joinpath(PATH,"CORRELATOR_NN_$(ens_name).pdf"))





## ===========================================================
    seed = 1994
    Tmax = 500
    δt   = 0.1
    Ns   = 10

    rng = Xoshiro(seed)
    Oav = mean(source.(cfgs))


    divf1 = 0.
    divf2 = [0.]

    cnt = 0
    out = map(cfgs) do z₀ 
        noise = [randn(rng,vol...,Tmax) for _ in 1:Ns]

        fk = z -> begin
            fout = map(noise) do η FK(z,η,δt) end |> stack 
            dmean(fout,dims=ndims(fout))
        end

        f = fk(z₀)

        divf1 = tr(ForwardDiff.jacobian(fk,z₀))
        divf2 = TRJ(rng, fk, z₀; ns=2)
        
        f∇S = grad_S(z₀) ⋅ f
        ΔO₀ = source(z₀) #- Oav

        global cnt += 1
        @show  cnt, divf1 - f∇S + ΔO₀, mean(divf2) - f∇S + ΔO₀

        (f,divf1,divf2)
    end
    
    fout = getindex.(out,1)
    trj1 = stack(getindex.(out,2))
    trj2 = getindex.(out,3) .|> mean |> stack

## --- Plot
    tag = ens_name*"2"

    # Cheat reweigthing
        # x = sumvol(ϕ,dims=Tuple(2:D))
        x = sumvol(stack(fout),dims=Tuple(2:D))
        corr2 = uwcorr(x[1:1,:] .* x,tag)
        corr2 ./= first(corr2)
        uwerr.(corr2)


    # Cheat reweigthing
        # x = sumvol(ϕ,dims=Tuple(2:D))
        x = sumvol(stack(fout),dims=Tuple(2:D))
        corr2 = uwcorr(x[1:1,:] .* x,tag)
        corr2 ./= first(corr2)
        uwerr.(corr2)

    # Improved reweighting (max uncoupled noise)
        ztld = map(zip(cfgs,fout)) do (z,f) @. z+ε*f end

        w = -(action.(ztld) .- ε .* source.(ztld)) .+ action.(cfgs)
        w .+= ε .* dmean(trj1,dims=1)
        w = exp.(w)
        w̃ = uwreal(w,tag)

        y = reshape(w,1,:) .* dsum(stack(ztld),dims=Tuple(2:D))
        corri = uwcorr(y,tag) ./ w̃

        corri = getindex.(corri,2)
        corri ./= first(corri)
        uwerr.(corri)



##
    off =  1e-3

    lam = round(λ,digits=2)
    kap = round(κ,digits=3)

    p = plot(
        formatter=:latex,framestyle=:box,
        ylim=(off,2),
        # xlim=(1,20),
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
        title=L"L=%$(first(vol))\times %$(last(vol)), \lambda=%$lam, \kappa=%$kap",
        titlefontsize=30
    )

    x = 1:first(vol)

    y,k = plottable(corr2; off=off)
    scatter!(p,x,y;k...,ms=7,c=jc.blue,msc=jc.blue,label="2pts")
    
    # y,k = plottable(corrw; off=off)
    # scatter!(p,x.-0.2,y;k...,ms=7,m=:utriangle,c=jc.green,msc=jc.green,label="FK")

    y,k = plottable(corri; off=off)
    scatter!(p,x.-0.2,y;k...,ms=7,m=:utriangle,c=jc.red,msc=jc.red,label="FK")



    display(p)
    # PATH = "/Users/pietro/Documents/Physics/Talks_and_presentations/2025-2026/QTC2026/"
    # savefig(p,joinpath(PATH,"CORRELATOR_NN_$(ens_name).pdf"))

