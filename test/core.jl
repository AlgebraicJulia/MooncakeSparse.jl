using MooncakeSparse
using MooncakeSparse: toarray
using Mooncake: Config, prepare_derivative_cache, prepare_gradient_cache,
                zero_tangent, value_and_derivative!!, value_and_gradient!!,
                Tangent
using LinearAlgebra
using SparseArrays
using SparseArrays: nonzeros
using Random: randn!
using Test

function randtangent(A)
    dA = copy(A)
    return randtangent!(dA)
end

function randtangent(A::Adjoint)
    return adjoint(randtangent(parent(A)))
end

function randtangent(A::Transpose)
    return transpose(randtangent(parent(A)))
end

function randtangent!(A::AbstractArray)
    return randn!(A)
end

function randtangent!(A::SparseMatrixCSC)
    randn!(nonzeros(A))
    return A
end

function randtangent!(A::Union{Hermitian,Symmetric})
    randtangent!(parent(A))
    return A
end

function maybefriendly(x, g)
    if g isa Tangent
        return toarray(x, g)
    end

    return g
end

function flat(A)
    return A
end

function flat(A::Union{Symmetric, Hermitian, Adjoint, Transpose})
    return flat(parent(A))
end

function dotflat(A, B)
    return dot(flat(A), flat(B))
end

function testadjoint(f, args...; rtol=1e-4)
    config = Config(friendly_tangents=true)

    fwd_cache = prepare_derivative_cache(f, args...; config)
    df = zero_tangent(f)

    tangents = map(randtangent, args)
    _, dy = value_and_derivative!!(fwd_cache, (f, df), zip(args, tangents)...)

    rev_cache = prepare_gradient_cache(f, args...; config)
    _, (_, gradients...) = value_and_gradient!!(rev_cache, f, args...)

    arrays = map(maybefriendly, args, gradients)
    return isapprox(dy, sum(real ∘ splat(dotflat), zip(arrays, tangents)); rtol)
end

n = 100
k = 10
p = 0.3

@testset "Adjoint Consistency" begin
    for T in (Float64, ComplexF64)
        @testset "$T" begin
            AU = sprandn(T, n, n, p) + n * I
            AH = Hermitian(AU, :L)
            AS = Symmetric(AU, :L)

            x = randn(T, n)
            y = randn(T, n)

            X = randn(T, k, n)
            Y = randn(T, n, k)

            @testset "lmul" begin
                for L in (AU, AU', transpose(AU), transpose(AU)', AH, AS, transpose(AH), AS'), R in (X', Y, y)
                    @test testadjoint((L, R) -> real(sum(L * R)), L, R)
                end
            end

            @testset "rmul" begin
                for L in (X, Y', y'), R in (AU, AU', transpose(AU), transpose(AU)')
                    @test testadjoint((L, R) -> real(sum(L * R)), L, R)
                end
            end

            @testset "dot (3-arg)" begin
                for A in (AU, AU', transpose(AU), transpose(AU)', AH, AS, transpose(AH), AS')
                    @test testadjoint((x, A, y) -> real(dot(x, A, y)), x, A, y)
                end
            end

            @testset "ldivwith" begin
                for A in (AU, AH, AS), R in (Y, y)
                    @test testadjoint((A, R) -> real(sum(ldivwith(A, lu(Matrix(A)), R))), A, R)
                end
            end

            # rdivwith tests disabled due to Julia stdlib bug:
            # mul!(C, Y, Hermitian{SparseMatrixCSC}) ignores the Hermitian wrapper
            # https://github.com/JuliaSparse/SparseArrays.jl/issues/688
        end
    end
end

