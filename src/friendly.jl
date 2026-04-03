function Mooncake.friendly_tangent_cache(x::MaybeHermOrSymSparse)
    return FriendlyTangentCache{AsCustomised}(similar(x))
end

function Mooncake.friendly_tangent_cache(x::Union{AdjSparse, ConjSymSparse})
    return FriendlyTangentCache{AsCustomised}(adjoint(similar(parent(x))))
end

function Mooncake.friendly_tangent_cache(x::Union{TransSparse, ConjHermSparse})
    return FriendlyTangentCache{AsCustomised}(transpose(similar(parent(x))))
end

function Mooncake.friendly_tangent_cache(x::Adjoint{T, TransSparse{T, I}}) where {T, I}
    return FriendlyTangentCache{AsCustomised}(adjoint(transpose(similar(parent(parent(x))))))
end

function Mooncake.friendly_tangent_cache(x::Transpose{T, AdjSparse{T, I}}) where {T, I}
    return FriendlyTangentCache{AsCustomised}(transpose(adjoint(similar(parent(parent(x))))))
end

function Mooncake.tangent_to_friendly_internal!!(
        dest::SparseMatrixCSC{T, I},
        ::SparseMatrixCSC{T, I},
        tangent,
    ) where {T, I}
    copyto!(nonzeros(dest), tangentdata(tangent).nzval)
    return dest
end

function Mooncake.tangent_to_friendly_internal!!(
        dest::HermOrSymSparse{T, I},
        ::HermOrSymSparse{T, I},
        tangent,
    ) where {T, I}
    copyto!(nonzeros(parent(dest)), tangentdata(tangent).data.fields.nzval)
    return dest
end

function Mooncake.tangent_to_friendly_internal!!(
        dest::AdjOrTransSparse{T, I},
        ::AdjOrTransSparse{T, I},
        tangent,
    ) where {T, I}
    copyto!(nonzeros(parent(dest)), tangentdata(tangent).parent.fields.nzval)
    return dest
end

function Mooncake.tangent_to_friendly_internal!!(
        dest::ConjHermOrSymSparse{T, I},
        ::ConjHermOrSymSparse{T, I},
        tangent,
    ) where {T, I}
    copyto!(nonzeros(parent(parent(dest))), tangentdata(tangent).parent.fields.data.fields.nzval)
    return dest
end

function Mooncake.tangent_to_friendly_internal!!(
        dest::ConjSparse{T, I},
        ::ConjSparse{T, I},
        tangent,
    ) where {T, I}
    copyto!(nonzeros(parent(parent(dest))), tangentdata(tangent).parent.fields.parent.fields.nzval)
    return dest
end
