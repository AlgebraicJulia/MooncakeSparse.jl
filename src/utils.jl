function tangentdata(x::Tangent)
    return tangentdata(x.fields)
end

function tangentdata(x::NamedTuple{(:fields,)})
    return tangentdata(x.fields)
end

function tangentdata(x::NamedTuple)
    return x
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

function unwrapsym(A::SymSparse)
    return parent(A), Val(:N), Val(:N), A.uplo
end

function unwrapsym(A::HermSparse)
    return parent(A), Val(:C), Val(:N), A.uplo
end

function unwrapsym(A::ConjSymSparse)
    S = parent(A)
    return parent(S), Val(:N), Val(:C), S.uplo
end

function unwrapsym(A::ConjHermSparse)
    H = parent(A)
    return parent(H), Val(:C), Val(:C), H.uplo
end

function intriangle(uplo, i::Integer, j::Integer)
    return isnothing(uplo) || (uplo == 'L' && i >= j) || (uplo == 'U' && i <= j)
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
    return selupd_impl!(C, AP, BP, α, β, tA, cA, tB, cB, uplo)
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

function selupd_impl!(C::SparseMatrixCSC, A::AbstractVector, B::AbstractVector, α, β, ::Val{tA}, ::Val{cA}, ::Val{tB}, ::Val{cB}, uplo=nothing) where {tA, cA, tB, cB}
    @assert size(C, 1) == size(C, 2) == length(A) == length(B)

    @inbounds for j in axes(C, 2)
        if cB === :C
            Bj = conj(B[j])
        else
            Bj = B[j]
        end

        for p in nzrange(C, j)
            i = rowvals(C)[p]

            if intriangle(uplo, i, j)
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

function selupd_impl!(C::SparseMatrixCSC, A::AbstractMatrix, B::AbstractMatrix, α, β, tA::Val{TA}, cA::Val{CA}, tB::Val{TB}, cB::Val{CB}, uplo=nothing) where {TA, CA, TB, CB}
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

        selupd_impl!(C, Ak, Bk, α, true, tA, cA, tB, cB, uplo)
    end

    return C
end

function ensureuplo(A::SparseMatrixCSC, tB::Val{TB}, src::Char, tgt::Char) where TB
    if src == tgt
        return A
    elseif TB === :N
        return copy(transpose(A))
    else
        return copy(adjoint(A))
    end
end

function selaxpby!(α, B, β, A::AdjSparse)
    selaxpby!(α, adjoint(B), conj(β), adjoint(A))
    return A
end

function selaxpby!(α, B, β, A::TransSparse)
    selaxpby!(α, transpose(B), β, transpose(A))
    return A
end

function selaxpby!(α, B::MaybeAdjOrTransOrConjSparse, β, A::SparseMatrixCSC)
    BP, tB, cB = unwrap(B)
    selaxpby_rec_rec!(α, BP, tB, cB, β, A)
    return A
end

function selaxpby!(α, B::HermOrSymOrConjSparse, β, A::SparseMatrixCSC)
    BP, tB, cB, uB = unwrapsym(B)
    selaxpby_sym_rec!(α, BP, tB, cB, uB, β, A)
    return A
end

function selaxpby!(α, B::MaybeAdjOrTransOrConjSparse, β, A::HermOrSymSparse)
    BP, tB, cB = unwrap(B)
    AP, tA, cA, uA = unwrapsym(A)
    selaxpby_rec_sym!(α, BP, tB, cB, β, AP, tA, cA, uA)
    return A
end

function selaxpby!(α, B::HermOrSymOrConjSparse, β, A::HermOrSymSparse)
    BP, tB, cB, uB = unwrapsym(B)
    AP, tA, cA, uA = unwrapsym(A)
    selaxpby_sym_sym!(α, BP, tB, cB, uB, β, AP, tA, cA, uA)
    return A
end

function selaxpby_rec_rec!(α, B, tB::Val{TB}, cB, β, A) where TB
    if TB === :T
        B = copy(transpose(B))
    end

    selaxpby_impl!(α, B, β, A, cB, cB)
    return A
end

function selaxpby_sym_rec!(α, B, tB::Val{TB}, cB, uB, β, A) where TB
    BL = ensureuplo(B, tB, uB, 'L')
    BU = ensureuplo(B, tB, uB, 'U')

    if TB === :C
        dB = Val(:R)
    else
        dB = cB
    end

    selaxpby_impl!(α, BL, β,    A, cB, dB,      'L')
    selaxpby_impl!(α, BU, true, A, cB, Val(:U), 'U')
    return A
end

function selaxpby_rec_sym!(α, B, tB::Val{TB}, cB, β, A, tA::Val{TA}, cA, uA) where {TB, TA}
    if TA === :N || TB === :N
        cX = cB
    else
        cX = cflip(cB)
    end

    if TA === :N
        BT = copy(transpose(B))
    else
        BT = copy(adjoint(B))
    end

    selaxpby_impl!(α / 2, B,  β,    A, cX, cX, uA)
    selaxpby_impl!(α / 2, BT, true, A, cX, cX, uA)
    return A
end

function selaxpby_sym_sym!(α, B, tB::Val{TB}, cB, uB, β, A, tA::Val{TA}, cA, uA) where {TB, TA}
    B = ensureuplo(B, tB, uB, uA)

    if TA === TB
        cX = cB

        if TA === :N
            dX = cB
        else
            dX = Val(:N)
        end
    else
        cX = Val(:R)
        dX = Val(:R)
    end

    selaxpby_impl!(α, B, β, A, cX, dX, uA)
    return A
end

function selaxpby_impl!(α, B::SparseMatrixCSC{TB, IB}, β, A::SparseMatrixCSC{TA, IA}, cB::Val{CB}, dB::Val{DB}, uplo=nothing) where {TA, IA, TB, IB, CB, DB}
    @assert size(B) == size(A)

    @inbounds for j in axes(A, 2)
        pa, pastop = getcolptr(A)[j], getcolptr(A)[j + 1] - one(IA)
        pb, pbstop = getcolptr(B)[j], getcolptr(B)[j + 1] - one(IB)

        while pa <= pastop && pb <= pbstop
            ia, ib = rowvals(A)[pa], rowvals(B)[pb]

            if ia == ib
                if intriangle(uplo, ia, j)
                    if ia == j
                        XB = DB
                    else
                        XB = CB
                    end

                    if XB !== :U
                        Bij = nonzeros(B)[pb]

                        if XB === :C
                            Bij = conj(Bij)
                        elseif XB === :R
                            Bij = real(Bij)
                        end 

                        if iszero(β)
                            nonzeros(A)[pa] = α * Bij
                        else
                            nonzeros(A)[pa] = α * Bij + β * nonzeros(A)[pa]
                        end
                    end
                end

                pa += one(IA)
                pb += one(IB)
            elseif ia < ib
                pa += one(IA)
            else
                pb += one(IB)
            end
        end
    end

    return A
end
