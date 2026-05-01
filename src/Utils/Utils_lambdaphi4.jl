# This functions are given a *single conf* in input

nn(ϕ::AbstractArray{T,2}, s) where T = 
    circshift(ϕ,(s,0)) .+ 
    circshift(ϕ,(0,s))
nn(ϕ::AbstractArray{T,4}, s) where T = 
    circshift(ϕ,(s,0,0,0)) .+ 
    circshift(ϕ,(0,s,0,0)) .+ 
    circshift(ϕ,(0,0,s,0)) .+ 
    circshift(ϕ,(0,0,0,s))

_action(λ,κ) = ϕ ->
    ϕ.^2 .+ λ .* (ϕ.^2 .- 1).^2 .+ -2κ .* ϕ .* nn(ϕ,-1) |> sum

_grad_S(λ,κ) = ϕ -> 
    2ϕ .+ 4λ .* (ϕ.^2 .- 1) .* ϕ .+ -2κ .* (nn(ϕ,-1) .+ nn(ϕ,1))

_Hvp(λ,κ) = (ϕ,v) ->         
    (2 .- 4λ .+ 12λ .* ϕ.^2) .* v .+ -2κ .* (nn(v,-1) .+ nn(v,1))

_fHf(λ,κ) = (ϕ,f) -> f ⋅ _Hvp(λ,κ)(ϕ,f) 


propagator(m2,vol) = begin
    p = 2π/first(vol)  .* collect((1:first(vol) ).-1)
    g0 = @. 1 / (  (2 * sin(p/2))^2   + m2 )

    G̃ = zeros(vol...);
    G̃[:,Int.(ones(length(vol)-1))...] .= g0

    real.(ifft(G̃));
end

propagator(κ,m2,vol) = propagator(m2,vol) .* prod(vol[2:end]) / 2κ

source(z; x₀=1) = sum(selectdim(z,1,x₀)) 