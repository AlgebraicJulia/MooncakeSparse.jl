using MooncakeSparse
using Mooncake: Config, prepare_derivative_cache, prepare_gradient_cache,
                zero_tangent, value_and_derivative!!, value_and_gradient!!
using LinearAlgebra
using SparseArrays
using SparseArrays: nonzeros
using Random: randn!

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

function testadjoint(f, args...; rtol=1e-4)
    config = Config(friendly_tangents=true)

    fwd_cache = prepare_derivative_cache(f, args...; config)
    df = zero_tangent(f)

    tangents = map(randtangent, args)
    _, dy = value_and_derivative!!(fwd_cache, (f, df), zip(args, tangents)...)

    rev_cache = prepare_gradient_cache(f, args...; config)
    _, (_, gradients...) = value_and_gradient!!(rev_cache, f, args...)

    return isapprox(dy, sum(real ∘ splat(dot), zip(gradients, tangents)); rtol)
end

n = 100
k = 10
p = 0.3

@testset "Adjoint Consistency" begin
    for T in (Float64, ComplexF64)
        @testset "$T" begin
            AU = sprandn(T, n, n, p)
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

            @testset "dot (2-arg)" begin
                BU = sprandn(T, n, n, p)
                BH = Hermitian(BU, :L)
                BS = Symmetric(BU, :L)

                for L in (AU, AU', transpose(AU), AH, AS)
                    for R in (AU, AU', transpose(AU), AH, AS)
                        @test testadjoint((L, R) -> real(dot(L, R)), L, R)
                    end
                end
            end
        end
    end
end

