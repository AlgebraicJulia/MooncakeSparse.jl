function lmul_fwd_impl!!(cdC, cdA, cdB)
    C, dC = primaltangent(cdC)
    A, dA = primaltangent(cdA)
    B, dB = primaltangent(cdB)

    mul!(dC, dA, B)
    mul!(dC, A, dB, true, true)

    mul!(C, A, B)

    return cdC
end

function rmul_fwd_impl!!(cdC, cdA, cdB)
    C, dC = primaltangent(cdC)
    A, dA = primaltangent(cdA)
    B, dB = primaltangent(cdB)

    mul!(dC, dA, B)
    mul!(dC, A, dB, true, true)

    mul!(C, A, B)

    return cdC
end

function dot_fwd_impl!!(cdx, cdA, cdy)
    x, dx = primaltangent(cdx)
    A, dA = primaltangent(cdA)
    y, dy = primaltangent(cdy)

    z = dot(x, A, y)
    dz = dot(dx, A, y) + dot(x, dA, y) + dot(x, A, dy)

    return Dual(z, dz)
end

function lmul_rev_impl!!(cdC, cdA, cdB)
    C, dC = primaltangent(cdC)
    A, dA = primaltangent(cdA)
    B, dB = primaltangent(cdB)

    D = copy(C)

    mul!(C, A, B)

    function pullback!!(::NoRData)
        mul!(dB, A', dC, true, true)
        selupd!(dA, dC, B', true, true)
        copyto!(C, D)

        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return cdC, pullback!!
end

function rmul_rev_impl!!(cdC, cdA, cdB)
    C, dC = primaltangent(cdC)
    A, dA = primaltangent(cdA)
    B, dB = primaltangent(cdB)

    D = copy(C)

    mul!(C, A, B)

    function pullback!!(::NoRData)
        mul!(dA, dC, B', true, true)
        selupd!(dB, A', dC, true, true)
        copyto!(C, D)

        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return cdC, pullback!!
end

function dot_rev_impl!!(cdx, cdA, cdy)
    x, dx = primaltangent(cdx)
    A, dA = primaltangent(cdA)
    y, dy = primaltangent(cdy)

    Ay = A * y
    Ax = A' * x
    z = dot(x, Ay)

    function pullback!!(Δz)
        axpy!(Δz, Ay, dx)
        axpy!(Δz, Ax, dy)
        selupd!(dA, x, y', Δz, 1)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(z, NoFData()), pullback!!
end

for ST in (SparseMatrixCSC, AdjSparse, TransSparse, ConjSparse)
    @eval @is_primitive MinimalCtx Tuple{typeof(mul!), DenseVecOrMat{T}, $ST{T, I}, DenseVecOrMat{T}} where {T, I}
    @eval Mooncake.frule!!(::Dual{typeof(mul!)}, cdC::Dual{<:DenseVecOrMat}, cdA::Dual{<:$ST}, cdB::Dual{<:DenseVecOrMat}) = lmul_fwd_impl!!(cdC, cdA, cdB)
    @eval Mooncake.rrule!!(::CoDual{typeof(mul!)}, cdC::CoDual{<:DenseVecOrMat}, cdA::CoDual{<:$ST}, cdB::CoDual{<:DenseVecOrMat}) = lmul_rev_impl!!(cdC, cdA, cdB)

    @eval @is_primitive MinimalCtx Tuple{typeof(mul!), DenseMatrix{T}, DenseMatrix{T}, $ST{T, I}} where {T, I}
    @eval Mooncake.frule!!(::Dual{typeof(mul!)}, cdC::Dual{<:DenseMatrix}, cdA::Dual{<:DenseMatrix}, cdB::Dual{<:$ST}) = rmul_fwd_impl!!(cdC, cdA, cdB)
    @eval Mooncake.rrule!!(::CoDual{typeof(mul!)}, cdC::CoDual{<:DenseMatrix}, cdA::CoDual{<:DenseMatrix}, cdB::CoDual{<:$ST}) = rmul_rev_impl!!(cdC, cdA, cdB)

    @eval @is_primitive MinimalCtx Tuple{typeof(dot), StridedVector{T}, $ST{T, I}, StridedVector{T}} where {T, I}
    @eval Mooncake.frule!!(::Dual{typeof(dot)}, cdx::Dual{<:StridedVector}, cdA::Dual{<:$ST}, cdy::Dual{<:StridedVector}) = dot_fwd_impl!!(cdx, cdA, cdy)
    @eval Mooncake.rrule!!(::CoDual{typeof(dot)}, cdx::CoDual{<:StridedVector}, cdA::CoDual{<:$ST}, cdy::CoDual{<:StridedVector}) = dot_rev_impl!!(cdx, cdA, cdy)
end
