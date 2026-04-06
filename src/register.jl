const SPARSE_MATRIX_TYPES = (
    SparseMatrixCSC, AdjSparse, TransSparse, ConjSparse,
    HermSparse, SymSparse, ConjHermSparse, ConjSymSparse,
)

for SL in SPARSE_MATRIX_TYPES
    @eval @is_primitive MinimalCtx Tuple{typeof(dot), StridedVector{T}, $SL{T, I}, StridedVector{T}} where {T, I}
    @eval Mooncake.frule!!(::Dual{typeof(dot)}, cdx::Dual{<:StridedVector}, cdA::Dual{<:$SL}, cdy::Dual{<:StridedVector}) = dot_fwd_impl!!(cdx, cdA, cdy)
    @eval Mooncake.rrule!!(::CoDual{typeof(dot)}, cdx::CoDual{<:StridedVector}, cdA::CoDual{<:$SL}, cdy::CoDual{<:StridedVector}) = dot_rev_impl!!(cdx, cdA, cdy)

    @eval @is_primitive MinimalCtx Tuple{typeof(mul!), DenseVecOrMat{T}, $SL{T, I}, DenseVecOrMat{T}} where {T, I}
    @eval Mooncake.frule!!(::Dual{typeof(mul!)}, cdC::Dual{<:DenseVecOrMat}, cdA::Dual{<:$SL}, cdB::Dual{<:DenseVecOrMat}) = lmul_fwd_impl!!(cdC, cdA, cdB)
    @eval Mooncake.rrule!!(::CoDual{typeof(mul!)}, cdC::CoDual{<:DenseVecOrMat}, cdA::CoDual{<:$SL}, cdB::CoDual{<:DenseVecOrMat}) = lmul_rev_impl!!(cdC, cdA, cdB)

    @eval @is_primitive MinimalCtx Tuple{typeof(mul!), DenseMatrix{T}, DenseMatrix{T}, $SL{T, I}} where {T, I}
    @eval Mooncake.frule!!(::Dual{typeof(mul!)}, cdC::Dual{<:DenseMatrix}, cdA::Dual{<:DenseMatrix}, cdB::Dual{<:$SL}) = rmul_fwd_impl!!(cdC, cdA, cdB)
    @eval Mooncake.rrule!!(::CoDual{typeof(mul!)}, cdC::CoDual{<:DenseMatrix}, cdA::CoDual{<:DenseMatrix}, cdB::CoDual{<:$SL}) = rmul_rev_impl!!(cdC, cdA, cdB)

    @eval @is_primitive MinimalCtx Tuple{typeof(ldivwith!), $SL, <:Any, AbstractVecOrMat}
    @eval Mooncake.frule!!(::Dual{typeof(ldivwith!)}, cdA::Dual{<:$SL}, cdF::Dual, cdX::Dual{<:AbstractVecOrMat}) = ldiv_fwd_impl!!(cdA, cdF, cdX)
    @eval Mooncake.rrule!!(::CoDual{typeof(ldivwith!)}, cdA::CoDual{<:$SL}, cdF::CoDual, cdX::CoDual{<:AbstractVecOrMat}) = ldiv_rev_impl!!(cdA, cdF, cdX)

    @eval @is_primitive MinimalCtx Tuple{typeof(rdivwith!), AbstractMatrix, $SL, <:Any}
    @eval Mooncake.frule!!(::Dual{typeof(rdivwith!)}, cdX::Dual{<:AbstractMatrix}, cdA::Dual{<:$SL}, cdF::Dual) = rdiv_fwd_impl!!(cdX, cdA, cdF)
    @eval Mooncake.rrule!!(::CoDual{typeof(rdivwith!)}, cdX::CoDual{<:AbstractMatrix}, cdA::CoDual{<:$SL}, cdF::CoDual) = rdiv_rev_impl!!(cdX, cdA, cdF)
end
