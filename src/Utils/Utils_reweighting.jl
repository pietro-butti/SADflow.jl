function compute_weights(z::AbstractArray{T,N},f::AbstractArray{T,N},act,src; divf=nothing) where {T,N}
    ε = Series((zero(T),one(T)))
    z̃ = z .+ ε .* f # transport map
    logw = -(act(z̃) .- ε .* src(z̃)) .+ act(z)
    if !isnothing(divf) logw .+= ε .* divf end

    return exp.(logw)
end
