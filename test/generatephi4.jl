using Pkg; Pkg.activate(".")
using SADflow
using Plots, Random, StatsBase
using Lux, NNlib, MLUtils, Optimisers
using Zygote

## ============= Function to compute the action =================
    staple_sum(ϕ::AbstractArray{T,2}) where T = 
        circshift(ϕ,(0,1)) .+ 
        circshift(ϕ,(1,0))


    staple_sum(ϕ::AbstractArray{T,4}) where T = 
        circshift(ϕ,(1,0,0,0)) .+ 
        circshift(ϕ,(0,1,0,0)) .+ 
        circshift(ϕ,(0,0,1,0)) .+ 
        circshift(ϕ,(0,0,0,1)) 

    Actionλϕ⁴(ϕ::AbstractArray{T,N},λ::T,κ::T) where {T,N} = (
        ϕ.^2 .+ λ .* (ϕ.^2 .- one(T)).^2 .+
        -2κ .* ϕ .* staple_sum(ϕ)
    ) |> x->sum(x; dims=1:N-1)
## ===============================================================

## ============= Loss function =================
    function inverseKL(model, ps, st, z::AbstractArray{T,N}, λ::T,κ::T) where {T,N}
        zero_batch = zeros_like(z, size(z,N))
        (ϕ, logdetJ), st = model((z, zero_batch), ps, st)

        logp = -Actionλϕ⁴(ϕ,λ,κ)[:]
        logr = sum(log_N01(z); dims=collect(1:N-1))[:]
        logq = logr - logdetJ

        logw = logp - logq
        log_ess = 2*logsumexp(logw) - logsumexp(2* logw)
        ess = exp(log_ess)/length(logw)

        return mean(logq - logp), st, (; ess=ess)
    end

    setloss(l,k) = (m,p,s,z) -> inverseKL(m,p,s,z,l,k)

## =============================================


    
##
    vol = (8,8)
    
    (λ,κ)  = (0.145f0, 0.3f0)
    iKL = setloss(λ,κ)
    
    η = 0.001f0
    batch_size = 100
    seed = 1994
    
##
    pad = WrappedFunction(x->pad_circular(x,(2,0,2,0)))
    net = Chain(
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1; use_bias=false),
        pad,Conv((3,3),1=>1, leakyrelu; use_bias=false),
        pad,Conv((3,3),1=>2, tanh; use_bias=false),
        x->biject(x; dim=3)
    )
    model = Chain(
        [AffineEOCoupling(net,(vol...,1),i%2==0) for i in 1:10]...
    )
    rng = Xoshiro(seed)
    ps, st = Lux.setup(rng,model)
##
    metrics = (hloss=[],ess=[])
    ts = Training.TrainState(model, ps, st, Optimisers.Adam(η));
##
    # ts = Training.TrainState(model, ts.parameters, ts.states, Optimisers.Adam(0.004f0));

    for epoch in 1:10000
        stime = time()
            z_sample = randn(rng,Float32,vol...,1,batch_size)
            _, loss, _, tstate = Training.single_train_step!( 
                AutoZygote(),  
                iKL, z_sample, ts
            );
        ttime = time()-stime

        l,_,s = iKL(model, ts.parameters, Lux.testmode(ts.states), z_sample)
        push!(metrics.hloss,l)
        push!(metrics.ess,s.ess)

        @show epoch, round(ttime,digits=3), l, s.ess
    end
##

plot(metrics.ess)
# plot(metrics.hloss,yscale=:log10)
plot(log.(metrics.hloss .- minimum(metrics.hloss)))


