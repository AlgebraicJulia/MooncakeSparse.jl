module MooncakeSparse

using LinearAlgebra
using LinearAlgebra: Hermitian, Symmetric, Adjoint, Transpose, AdjOrTrans, rmul!, dot, axpy!, axpby!, mul!

using SparseArrays
using SparseArrays: SparseMatrixCSC, nzrange, rowvals, getcolptr, nonzeros

using Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, Dual, NoRData, NoFData, primal, tangent,
                Tangent, FData

const HermSparse{T, I} = Hermitian{T, SparseMatrixCSC{T, I}}
const SymSparse{T, I} = Symmetric{T, SparseMatrixCSC{T, I}}
const HermOrSymSparse{T, I} = Union{HermSparse{T, I}, SymSparse{T, I}}

const ConjHermSparse{T, I} = Transpose{T, HermSparse{T, I}}
const ConjSymSparse{T, I} = Adjoint{T, SymSparse{T, I}}

const AdjSparse{T, I} = Adjoint{T, SparseMatrixCSC{T, I}}
const TransSparse{T, I} = Transpose{T, SparseMatrixCSC{T, I}}
const ConjSparse{T, I} = Union{Adjoint{T, TransSparse{T, I}}, Transpose{T, AdjSparse{T, I}}}

const DenseMatrix{T} = Union{StridedMatrix{T}, AdjOrTrans{T, <:StridedVecOrMat{T}}}
const DenseVecOrMat{T} = Union{DenseMatrix{T}, StridedVector{T}}

include("utils.jl")
include("symmetric.jl")
include("matrix.jl")

end # module MooncakeSparse
