# This one only works when last dimension is iterable

vjv = (rng, func, z)-> begin
    η = rand(rng,[-1,1],size(z)...) .|> eltype(z)
    Jη = jacobian_vector_product(func, AutoForwardDiff(), z, η)
    @assert ndims(η)==ndims(Jη)
    dsum(η .* Jη, dims=Tuple(1:ndims(η)-1))
end

trJ(rng, func, z; ns=1) = 
    @pipe [vjv(rng,func,z) for _ in 1:ns] |> stack |> dmean(_,dims=2)
trJ(func, z; ns=1) = 
    trJ(Random.default_rng(), func,z; ns=ns)

vjjv = (rng, func, z)-> begin
    η = rand(rng,[-1,1],size(z)...) .|> eltype(z)
    Jη  = jacobian_vector_product(func, AutoForwardDiff(), z, η)
    JJη = jacobian_vector_product(func, AutoForwardDiff(), z, Jη)
    @assert ndims(η)==ndims(JJη)
    dsum(η .* JJη, dims=Tuple(1:ndims(η)-1))
end

trJJ(rng, func, z; ns=1) = 
    @pipe [vjjv(rng,func,z) for _ in 1:ns] |> stack |> dmean(_,dims=2)
trJJ(func, z; ns=1) = 
    trJJ(Random.default_rng(), func,z; ns=ns)