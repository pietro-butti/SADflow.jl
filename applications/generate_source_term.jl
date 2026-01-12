using Pkg; Pkg.activate(".")
using SADflow
using Plots, Random, StatsBase, Printf
using Lux, NNlib, MLUtils, Optimisers
using Zygote
using Lattice

SEED = 1994
rng = Xoshiro(SEED)


## ============== Read configuration from ensemble ==============
    # folder = "/Users/pietro/code/julia/sketches/inverseRG/applications/2d_l0.01_k0.125_L_128_32"
    folder = "/Users/pietro/code/software/SADflow/.local_sketches/2d_l0.0_k0.12461_L_16_16"
    vol = (16,16)
    (κ,λ,α) = (0.12461f0,0.f0,1.f0)
    
    # Infer configuration indexes -----------------------------
        ens_name = split(folder,"/") |> last
        conflist = filter(x->endswith(x,".jld2"),readdir(folder))
        str_idx = map(x->split(x,".")[end-1][4:end], conflist)
        confidx = parse.(Int,str_idx)
    # Setup readers -------------------------------------------
        space = Grid{length(vol)}(vol,Tuple(8 for _ in vol))
        ws    = Phi4_workspace(space)
        
        suffix = i -> ".cfg$(lpad(i,5,'0')).jld2"

        pics = Array{Float32}(undef, space.iL...,1,0)
        for icfg in confidx
            Lattice.Fields.read!(ws._phi, "$folder/$ens_name$(suffix(icfg))")
            pics = cat(pics,conf_to_pic(ws._phi,space),dims=ndims(pics))
        end
    # ----------------------------------------------------------

    # ϕ = dropdims(pics; dims=3)
    ϕ = Float32.(pics)
## ===============================================================

## ================== Model definition ====================
    pad = WrappedFunction(x->pad_circular(x,(2,0,2,0)))
    net = Chain(
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1, leakyrelu; use_bias=false),
        pad,Conv((3,3),1=>2, tanh; use_bias=false),
        x->biject(x; dim=3)
    )
    model = Chain(
        [AffineEOCoupling(net,(vol...,1),i%2==0) for i in 1:10]...
    )
    ps, st = Lux.setup(rng,model)
## =======================================================


## ====================== Training ======================
    # Metaparameters -----------
        J = 0.01f0
        x₀ = 1
        η = 0.005f0
        BSIZE = 32
    # --------------------------

    LOSS = setloss(λ,κ,J)
    NSAMPLES = last(size(ϕ))

##
    metrics = (hloss=[],ess=[])
    ts = Training.TrainState(model, ps, st, Optimisers.Adam(η));
##
    zin = ϕ[:,:,:,1:BSIZE]
    LOSS(model,ps,st,zin)

    _, loss, metr, ts = @time Training.single_train_step!( 
        AutoZygote(),  
        LOSS, zin, ts;
    );

##
    for epoch in 1:100
        stime = time()
        
        for i in 1:NSAMPLES÷BSIZE
            i_samples = sample(rng,1:NSAMPLES,BSIZE,ordered=true)
            ϕ_sample = ϕ[:,:,:,i_samples]

            _, loss, mtr, ts = Training.single_train_step!( 
                AutoZygote(),  
                LOSS, ϕ_sample, ts;
            );
        end

        ttime = time()-stime

        l1,_,s1 = LOSS(model,ts.parameters, Lux.testmode(ts.states), ϕ)
        @printf("epoch=%i [%.3f s]: %.7f %.7f\n",epoch, ttime, l1, s1.ess)
        push!(metrics.hloss,l1)
        push!(metrics.ess,s1.ess)
    end
##

plot(metrics.ess)
# plot(metrics.hloss,yscale=:log10)
plot(log.(metrics.hloss .- minimum(metrics.hloss)))


