# --- Broadcasting --------------------------
import Base: length, iterate
length(x::uwreal) = 1
iterate(x::uwreal) = (x, nothing)
iterate(x::uwreal,::Nothing) = nothing

# --- matrix of C[time,conf] ----------------
uwcorr(x::Matrix,uwargs...) = 
    map(eachrow(x)) do r uwreal(r[:],uwargs...) end
uwcorr(x::AbstractArray{T,2},uwargs...) where T = 
    map(eachrow(x)) do r uwreal(r[:],uwargs...) end

# --- Formal series --------------------------
uwreal(x::Vector{Series{T,N}},uw...) where {T,N} = begin
    vecs = Tuple(getindex.(x,i) for i in 1:N)
    Series{uwreal,N}(Tuple(uwreal(v,uw...) for v in vecs))
end
import ADerrors.uwerr
uwerr(X::Series{uwreal,N}) where N = for i in 1:N uwerr(X[i]) end
uwerr(::Missing) = 0


uwerr(X::Series{uwreal,N},wpm...) where N = for i in 1:N uwerr(X[i],wmp...) end
uwerr(::Missing,wmp...) = 0

# --- Missing value for plotting -------------
import ADerrors: err, value
ADerrors.err(::Missing) = 0
ADerrors.value(::Missing) = missing