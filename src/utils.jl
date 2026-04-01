function tangentdata(x::Tangent)
    return x.fields
end

function tangentdata(x::FData)
    return x.data
end

function toarray(X::AbstractArray, dX)
    return dX
end

function toarray(X::SparseMatrixCSC, dX)
    dnzval = tangentdata(dX).nzval
    return SparseMatrixCSC(X.m, X.n, X.colptr, X.rowval, dnzval)
end

function toarray(X::Adjoint, dX)
    return adjoint(toarray(parent(X), tangentdata(dX).parent))
end

function toarray(X::Transpose, dX)
    return transpose(toarray(parent(X), tangentdata(dX).parent))
end

function toarray(X::Hermitian, dX)
    dY = toarray(parent(X), tangentdata(dX).data)
    return Hermitian(dY, Symbol(X.uplo))
end

function toarray(X::Symmetric, dX)
    dY = toarray(parent(X), tangentdata(dX).data)
    return Symmetric(dY, Symbol(X.uplo))
end

function primaltangent(x::CoDual)
    return (primal(x), tangent(x))
end

function primaltangent(x::CoDual{<:AbstractArray})
    X = primal(x)
    dX = tangent(x)
    return (X, toarray(X, dX))
end

function primaltangent(x::Dual)
    return (primal(x), tangent(x))
end

function primaltangent(x::Dual{<:AbstractArray})
    X = primal(x)
    dX = tangent(x)
    return (X, toarray(X, dX))
end

function tflip(::Val{:N})
    return Val(:T)
end

function tflip(::Val{:T})
    return Val(:N)
end

function cflip(::Val{:N})
    return Val(:C)
end

function cflip(::Val{:C})
    return Val(:N)
end

function unwrap(A::AbstractArray, tA::Val=Val(:N), cA::Val=Val(:N))
    return A, tA, cA
end

function unwrap(A::Transpose, tA::Val, cA::Val)
    return unwrap(transpose(A), tflip(tA), cA)
end

function unwrap(A::Adjoint, tA::Val, cA::Val)
    return unwrap(A', tflip(tA), cflip(cA))
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← (α/2) A B + (conj(α)/2) Bᴴ Aᴴ + β C
#
# The update is only applied to the structural nonzeros of C.
function selupd!(C::HermSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), C.uplo,         A,          B,       α  / 2, β)
    selupd!(parent(C), C.uplo, adjoint(B), adjoint(A), conj(α) / 2, true)
    return C
end

function selupd!(C::ConjHermSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), adjoint(transpose(A)), adjoint(transpose(B)), conj(α), conj(β))
    return C
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← (α/2) A B + (α/2) Bᵀ Aᵀ + β C
#
# The update is only applied to the structural nonzeros of C.
function selupd!(C::SymSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), C.uplo,           A,            B,  α / 2, β)
    selupd!(parent(C), C.uplo, transpose(B), transpose(A), α / 2, true)
    return C
end

function selupd!(C::ConjSymSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), adjoint(transpose(A)), adjoint(transpose(B)), conj(α), conj(β))
    return C
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← α A B + β C
#
# The update is only applied to the structural nonzeros of C
function selupd!(C::SparseMatrixCSC, uplo::Char, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    AP, tA, cA = unwrap(A)
    BP, tB, cB = unwrap(B)
    return selupd_impl!(C, uplo, AP, BP, α, β, tA, cA, tB, cB)
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← α A B + β C
#
# The update is only applied to the structural nonzeros of C
function selupd!(C::SparseMatrixCSC, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    AP, tA, cA = unwrap(A)
    BP, tB, cB = unwrap(B)
    return selupd_impl!(C, AP, BP, α, β, tA, cA, tB, cB)
end

function selupd!(C::AdjSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), adjoint(B), adjoint(A), conj(α), conj(β))
    return C
end

function selupd!(C::TransSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), transpose(B), transpose(A), α, β)
    return C
end

function selupd!(C::ConjSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(parent(C)), adjoint(transpose(A)), adjoint(transpose(B)), conj(α), conj(β))
    return C
end

function selupd_impl!(C::SparseMatrixCSC, A::AbstractVector, B::AbstractVector, α, β, ::Val{tA}, ::Val{cA}, ::Val{tB}, ::Val{cB}) where {tA, cA, tB, cB}
    @assert size(C, 1) == length(A)
    @assert size(C, 2) == length(B)

    @inbounds for j in axes(C, 2)
        if cB === :C
            Bj = conj(B[j])
        else
            Bj = B[j]
        end

        for p in nzrange(C, j)
            i = rowvals(C)[p]

            if cA === :C
                Ai = conj(A[i])
            else
                Ai = A[i]
            end

            if iszero(β)
                nonzeros(C)[p] = α * Ai * Bj
            else
                nonzeros(C)[p] = β * nonzeros(C)[p] + α * Ai * Bj
            end
        end
    end

    return C
end

function selupd_impl!(C::SparseMatrixCSC, A::AbstractMatrix, B::AbstractMatrix, α, β, tA::Val{TA}, cA::Val{CA}, tB::Val{TB}, cB::Val{CB}) where {TA, CA, TB, CB}
    if TA === :N && TB === :N
        @assert size(C, 1) == size(A, 1)
        @assert size(C, 2) == size(B, 2)
        @assert size(A, 2) == size(B, 1)
    elseif TA === :N && TB !== :N
        @assert size(C, 1) == size(A, 1)
        @assert size(C, 2) == size(B, 1)
        @assert size(A, 2) == size(B, 2)
    elseif TA !== :N && TB === :N
        @assert size(C, 1) == size(A, 2)
        @assert size(C, 2) == size(B, 2)
        @assert size(A, 1) == size(B, 1)
    else
        @assert size(C, 1) == size(A, 2)
        @assert size(C, 2) == size(B, 1)
        @assert size(A, 1) == size(B, 2)
    end

    if TA === :N
        rng = axes(A, 2)
    else
        rng = axes(A, 1)
    end

    if iszero(β)
        fill!(nonzeros(C), β)
    else
        rmul!(nonzeros(C), β)
    end

    for k in rng
        if TA === :N
            Ak = view(A, :, k)
        else
            Ak = view(A, k, :)
        end

        if TB === :N
            Bk = view(B, k, :)
        else
            Bk = view(B, :, k)
        end

        selupd_impl!(C, Ak, Bk, α, true, tA, cA, tB, cB)
    end

    return C
end

function selupd_impl!(C::SparseMatrixCSC, uplo::Char, A::AbstractVector, B::AbstractVector, α, β, ::Val{tA}, ::Val{cA}, ::Val{tB}, ::Val{cB}) where {tA, cA, tB, cB}
    @assert size(C, 1) == size(C, 2) == length(A) == length(B)

    @inbounds for j in axes(C, 2)
        if cB === :C
            Bj = conj(B[j])
        else
            Bj = B[j]
        end

        for p in nzrange(C, j)
            i = rowvals(C)[p]

            if (uplo == 'L' && i >= j) || (uplo == 'U' && i <= j)
                if cA === :C
                    Ai = conj(A[i])
                else
                    Ai = A[i]
                end

                if iszero(β)
                    nonzeros(C)[p] = α * Ai * Bj
                else
                    nonzeros(C)[p] = β * nonzeros(C)[p] + α * Ai * Bj
                end
            end
        end
    end

    return C
end

function selupd_impl!(C::SparseMatrixCSC, uplo::Char, A::AbstractMatrix, B::AbstractMatrix, α, β, tA::Val{TA}, cA::Val{CA}, tB::Val{TB}, cB::Val{CB}) where {TA, CA, TB, CB}
    @assert size(C, 1) == size(C, 2)

    if TA === :N && TB === :N
        @assert size(A, 1) == size(C, 1)
        @assert size(B, 2) == size(C, 1)
        @assert size(A, 2) == size(B, 1)
    elseif TA === :N && TB !== :N
        @assert size(A, 1) == size(C, 1)
        @assert size(B, 1) == size(C, 1)
        @assert size(A, 2) == size(B, 2)
    elseif TA !== :N && TB === :N
        @assert size(A, 2) == size(C, 1)
        @assert size(B, 2) == size(C, 1)
        @assert size(A, 1) == size(B, 1)
    else
        @assert size(A, 2) == size(C, 1)
        @assert size(B, 1) == size(C, 1)
        @assert size(A, 1) == size(B, 2)
    end

    if TA === :N
        rng = axes(A, 2)
    else
        rng = axes(A, 1)
    end

    if iszero(β)
        fill!(nonzeros(C), β)
    else
        rmul!(nonzeros(C), β)
    end

    for k in rng
        if TA === :N
            Ak = view(A, :, k)
        else
            Ak = view(A, k, :)
        end

        if TB === :N
            Bk = view(B, k, :)
        else
            Bk = view(B, :, k)
        end

        selupd_impl!(C, uplo, Ak, Bk, α, true, tA, cA, tB, cB)
    end

    return C
end
