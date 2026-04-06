function ldivwith!(A::AbstractMatrix, F, X::AbstractVecOrMat)
    ldiv!(F, X)
    return X
end

function ldivwith!(Y::AbstractVecOrMat, A::AbstractMatrix, F, X::AbstractVecOrMat)
    copyto!(Y, X)
    ldivwith!(A, F, Y)
    return Y
end

"""
    ldivwith(A, F, B)

Solve AX = B using a factorization F.
"""
function ldivwith(A::AbstractMatrix, F, X::AbstractVecOrMat)
    Y = similar(X, promote_type(eltype(F), eltype(X)))
    return ldivwith!(Y, A, F, X)
end

"""
    rdivwith(B, A, F)

Solve XA = B using a factorization F.
"""
function rdivwith!(X::AbstractMatrix, A::AbstractMatrix, F)
    rdiv!(X, F)
    return X
end

function rdivwith!(Y::AbstractMatrix, X::AbstractMatrix, A::AbstractMatrix, F)
    copyto!(Y, X)
    rdivwith!(Y, A, F)
    return Y
end

function rdivwith(X::AbstractMatrix, A::AbstractMatrix, F)
    Y = similar(X, promote_type(eltype(F), eltype(X)))
    return rdivwith!(Y, X, A, F)
end

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

    z = dot(x, A, y)

    function pullback!!(Δz)
        mul!(dx, A,  y, Δz, true)
        mul!(dy, A', x, Δz, true)
        selupd!(dA, x, y', Δz, true)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(z, NoFData()), pullback!!
end

function ldiv_fwd_impl!!(cdA, cdF, cdX)
    A, dA = primaltangent(cdA)
    X, dX = primaltangent(cdX)
    F = primal(cdF)

    ldivwith!(A, F, X)
    mul!(dX, dA, X, -1, true)
    ldivwith!(A, F, dX)

    return cdX
end

function ldiv_rev_impl!!(cdA, cdF, cdX)
    A, dA = primaltangent(cdA)
    X, dX = primaltangent(cdX)
    F = primal(cdF)
    Y = copy(X)

    ldivwith!(A, F, X)

    function pullback!!(::NoRData)
        ldivwith!(A', F', dX)
        selupd!(dA, dX, X', -1, true)
        copyto!(X, Y)
        return NoRData(), NoRData(), zero_rdata(F), NoRData()
    end

    return cdX, pullback!!
end

function rdiv_fwd_impl!!(cdX, cdA, cdF)
    X, dX = primaltangent(cdX)
    A, dA = primaltangent(cdA)
    F = primal(cdF)

    rdivwith!(X, A, F)
    mul!(dX, X, dA, -1, true)
    rdivwith!(dX, A, F)

    return cdX
end

function rdiv_rev_impl!!(cdX, cdA, cdF)
    X, dX = primaltangent(cdX)
    A, dA = primaltangent(cdA)
    F = primal(cdF)
    Y = copy(X)

    rdivwith!(X, A, F)

    function pullback!!(::NoRData)
        rdivwith!(dX, A', F')
        selupd!(dA, X', dX, -1, true)
        copyto!(X, Y)
        return NoRData(), NoRData(), zero_rdata(F), NoRData()
    end

    return cdX, pullback!!
end

