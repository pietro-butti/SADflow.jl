using JLD2, Pipe, StatsBase, Random, FFTW, Plots
using ADerrors, FormalSeries
using Lattice, SADflow

## =================== Load pre-generated configurations ===================
    ens_name = "2d_l0.0_k0.12461_L_32_8"
    data = load(".local_sketches/$ens_name.jld2")
    (vol,λ,κ) = data["vol"], data["λ"], data["κ"]
    ϕ = data["ϕ"][:,:,1,1:1000]
## ==========================================================================
    der(x::Series; ord=2) = getindex(x,ord)

    action = ϕ ->
        ϕ.^2 .+ λ .* (ϕ.^2 .- 1).^2 .+
        -2κ .* ϕ .* (
            circshift(ϕ,(-1,0)) .+ circshift(ϕ,(0,-1))
        ) |> sum

    Obs(ϕ::AbstractArray{T,2}; x₀=1) where T = @pipe sum(ϕ,dims=2) |> selectdim(_,1,x₀) |> only

    function reweight_corr(z,f,action,Obs,trJf; ens_name="ciao")
        cfgs = eachslice(z,dims=3)
        
        # Compute improved confs
        ε = Series((0.,1.))
        ϕ̃ = z .+ ε .* f
        phit = eachslice(ϕ̃,dims=3)

        # Compute weights
        ΔS = action.(phit) .- action.(cfgs)
        Õ = Obs.(phit)

        logw̃ = -ΔS .+ ε .* Õ .+ ε .* trJf
        w̃ = exp.(logw̃)

        # Reweight 1-pt function
        wf = @pipe reshape(w̃,(1 for _ in vol)...,:) .* ϕ̃ |> dsum(_,dims=2)

        # Compute correlator and error
        B = map(eachrow(der.(wf))) do r uwreal(collect(r),ens_name) end
        w₀ = uwreal(der.(w̃, ord=1), ens_name); uwerr(w₀)
        w₁ = uwreal(der.(w̃, ord=2), ens_name); uwerr(w₁)

        [(b - w₁)/w₀ for b in B]
    end
## ==========================================================================


    m2 = 1/κ - 2*length(vol)
    Lt = first(vol)
    Lx = last(vol)
    fp = [1 / 2 / 4 / ((2*sin(p/2))^2 + m2) for p in 2π/Lt.*(0:Lt-1)] 

    plot(fp)
    plot(fft(fp) .|> real, yscale=:log10)

    Fp = zeros(vol...)
    Fp[:,1] = fp      # rows = temporal, first column = zero spatial momentum ✓

    heatmap(Fp)
    f_free = real.(ifft(Fp))  # was fft
    # heatmap(f_free)

[pt*px for pt in 2π/Lt.*(0:Lt-1) for px in 2π/Lx.*(0:Lx-1)]




##

B = size(ϕ,ndims(ϕ))
corr = reweight_corr(ϕ,f_free,action,Obs,zeros(B); ens_name=ens_name)
corr .|> uwerr

ymin = corr .|> value .|>  abs |> minimum
scatter(value.(corr),yerror=ADerrors.err.(corr),label="1pts",yscale=:log10,ylim=(ymin/2,10))


corr2 = sum(ϕ[1:1,:,:] .* ϕ,dims=2)[:,1,:]
corr2 = map(eachrow(corr2)) do x uwreal(collect(x),ens_name) end
uwerr.(corr2)
scatter!((1:length(corr2)).+0.2,value.(corr2),yerror=ADerrors.err.(corr2),label="2pts")

