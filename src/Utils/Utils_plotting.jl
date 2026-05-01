meff(y::AbstractArray{T,1}) where T<:AbstractFloat = begin
    x = ((circshift(y,1) .+ circshift(y,-1))./2y)[2:end-1]
    acosh.(ifelse.(x .≥ 1,x,missing))
end
meff(y::Array{uwreal,1}) = begin
    x = ((circshift(y,1) .+ circshift(y,-1))./2y)[2:end-1]
    acosh.(ifelse.(value.(x) .≥ 1,x,missing))
end

import Base.abs
abs(x::uwreal) = ifelse(value(x)>0,x,-x)

function plottable(v::T,e::T; off=0, factor=0.5) where {T<:AbstractFloat}
    u = v .+ e
    d = v .- e

    if u<off
        y  = missing
        ye = (0.,0.)
    elseif u>off && v<off
        y = off*factor
        ye = (0.,u)
    elseif v>off && d<off
        y  = v
        ye = (v-off,e)
    else
        y  = v
        ye = (e,e)
    end

    return (y,ye)
end
plottable(uw::uwreal; kwargs...) = plottable(value(uw),ADerrors.err(uw); kwargs...)
plottable(uv::Vector{uwreal}; kwargs...) = begin
    out = plottable.(uv; kwargs...)
    getindex.(out,1), (; yerror=getindex.(out,2)) 
end