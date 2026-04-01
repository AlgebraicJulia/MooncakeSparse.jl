# MooncakeSparse.jl

[![CI/CD](https://github.com/AlgebraicJulia/MooncakeSparse.jl/actions/workflows/julia_ci.yml/badge.svg)](https://github.com/AlgebraicJulia/MooncakeSparse.jl/actions/workflows/julia_ci.yml)

Mooncake AD rules for sparse matrix operations.

## Supported Operations

- `mul!(C, A, B)` where `A` is sparse (lmul)
- `mul!(C, A, B)` where `B` is sparse (rmul)
- `dot(x, A, y)` where `A` is sparse

Supports `SparseMatrixCSC`, `Hermitian`, `Symmetric`, and their adjoints/transposes.

## Usage

```julia
using MooncakeSparse
using Mooncake
using SparseArrays

A = sprandn(100, 100, 0.1)
x = randn(100)
y = randn(100)

f(A, x, y) = dot(x, A, y)

cache = prepare_gradient_cache(f, A, x, y)
val, grads = value_and_gradient!!(cache, f, A, x, y)
```
