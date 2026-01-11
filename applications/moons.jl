using Pkg; Pkg.activate(".")
using SADflow
using Plots, Random, StatsBase
using Lux, Optimisers
# using Enzyme, Reactant
using Zygote

using Revise

## taken from https://lux.csail.mit.edu/stable/tutorials/intermediate/7_RealNVP
function make_moons(rng::AbstractRNG, ::Type{T}, n_samples::Int=100; noise::Union{Nothing,AbstractFloat}=nothing,) where {T}
    n_moons = n_samples ÷ 2
    t_min, t_max = T(0), T(π)
    t_inner = rand(rng, T, n_moons) * (t_max - t_min) .+ t_min
    t_outer = rand(rng, T, n_moons) * (t_max - t_min) .+ t_min
    outer_circ_x = cos.(t_outer)
    outer_circ_y = sin.(t_outer) .+ T(1)
    inner_circ_x = 1 .- cos.(t_inner)
    inner_circ_y = 1 .- sin.(t_inner) .- T(1)

    data = [outer_circ_x outer_circ_y; inner_circ_x inner_circ_y]
    z = permutedims(data, (2, 1))
    noise !== nothing && (z .+= T(noise) * randn(rng, T, size(z)))
    return z
end

rng = Xoshiro(1994)
zdata = make_moons(rng, Float32, 10_000; noise=0.1)


scatter(zdata[1,:],zdata[2,:],label="training set")



## Define model
    net = Chain(
        Dense(2=>16, gelu),
        [Dense(16=>16, gelu) for _ in 1:4]...,
        Dense(16=>4),
        WrappedFunction(biject)
    )

    model = Chain(
        [AffineEOCoupling(net, (2,), isodd(i)) for i in 1:10]...
    )
    ps, st = Lux.setup(rng,model)

## instantiate one flow model (w/ Reactant.jl)
    # const xdev = reactant_device()
    # xps, xst = xdev((ps,st))
    # xin = (zdata, zeros(size(zdata,2))) |> xdev
    # model_compiled = @compile model(xin, xps, Lux.testmode(xst))

## single train step (run this to see if everything compiles properly)
    tstate = Training.TrainState(model, ps, st, Optimisers.Adam(0.004f0))
    @time Training.single_train_step!(AutoZygote(), density_loss, zdata,tstate);


##
    hloss = []
    tstate = Training.TrainState(model, ps, st, Optimisers.Adam(0.0001f0))
    # tstate = Training.TrainState(model, tstate.parameters, tstate.states, Optimisers.Adam(0.001f0))
## training
    Ndata = size(zdata) |> last
    batch_size = 1000

    for epoch in 1:1000
        stime = time()
        for _ in 1:(Ndata÷batch_size)
            xtrain = zdata[:,sample(1:Ndata,batch_size)]

            _, loss, _, tstate = Training.single_train_step!(
                AutoZygote(), density_loss, xtrain, tstate
            )
        end
        ttime = time()-stime

        l,_,_ = density_loss(model, tstate.parameters, Lux.testmode(tstate.states), zdata)
        push!(hloss,l)
        @show epoch, round(ttime,digits=3), l 
    end
##

    plot(log.(hloss .- minimum(hloss)),xlabel="epochs of Adam(0.001) opt",label="log(L - min(L))")

##

    fl = StatefulLuxLayer(model,tstate.parameters,tstate.states)
    
    xin = (zdata,zeros(last(size(zdata))))
    out,_ = fl(xin)
    
    
    scatter(out[1,:],out[2,:],ms=.5,label="flowed training set")
    scatter!(zdata[1,:],zdata[2,:],label="data set")
    
    
    _sampler = pass_inverse(model, tstate.parameters, tstate.states)
    (xx,_) = _sampler(randn(Float32,(2,10_000)),zeros(10_000))


    scatter(xx[1,:],xx[2,:], label="sampled from N(0,1)")
    scatter!(zdata[1,:],zdata[2,:],ms=.5, label="training")
