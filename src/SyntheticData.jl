module SyntheticData
import LinearAlgebra

@doc raw"""
    rn_gen(μ::T, σ::T; N::Int64=1000,data::Bool=false) where T<:Real

It generates `N` random number normally distributed around `μ` with standard deviation `σ`
It returns mean and variance of the random numbers.
If `data` is set to `true`, it return also the random numbers
"""
function rn_gen(mu::T,sigma::T;N=1000, data=false) where T
    aux = randn(T,N)
    for i in eachindex(aux)
        aux[i] = aux[i]*sigma+mu;
    end
    mean =  sum(aux)/N;
    var = sum((aux[i]-mean)^2 for i in eachindex(aux))/(N-1);
    return data ? (mean,var,aux) : (mean,var)
end

@doc raw"""
    corr_data_gen(mu::AbstractVector, cov::AbstractMatrix;N::Int=1000, data=false)

generates correlated data using a multivariate distribution with true mean value `mu` and true covariance matrix `cov`.

It return the mean and covariance matrix of the generated data. if `data` is set to `true`, it return also the generated data
"""
function cov_data_gen(mu, cov::Matrix; data=false, N::Int=1000)
    if size(cov,1)!=length(mu)
        error("dimension mismatch: the correlation matrix and the true_value vector have different dimensions")
    end

    L = LinearAlgebra.cholesky(cov).L;
    aux = randn(length(mu),N)
    temp = similar(mu)

    for j in axes(aux,2)
        LinearAlgebra.mul!(temp,L,view(aux,:,j))
        aux[:,j] .= temp .+mu
    end
    mean = sum(aux,dims=2)./N
    C= zeros(length(mu),length(mu))
    for i in eachindex(mu), j in eachindex(mu)
         C[i,j]=sum((aux[i,t]-mean[i])*(aux[j,t]-mean[j]) for t in 1:N)/(N-1)
    end

    return data ? (mean,C,aux) : (mean,C)
end

@doc raw"""
    random_walk(mean::Float64; sigma::Float64 = 1.0, N::Int64=1000)
    random_walk(mean::Vector{Float64}; sigma::Union{Vector{Float64}, Float64}=1.0, kwargs...)

it generates a random walk around `mean`.

"""
function random_walk(mean::Float64; sigma::Float64=1.0, N::Int64=1000)
    x = zeros(N);
    x[1] = mean + randn()*sigma
    for i in 2:N
        x[i] = x[i-1] + randn()*sigma
        if abs(x[i] - mean) > sigma
            x[i] = x[i-1]
        end
    end
    return x
end

function random_walk(mean::Vector{Float64}; sigma::Union{Vector{Float64}, Float64}=1.0, kwargs...)
  return length(sigma) == 1 ? random_walk.(mean, sigma=sigma; kwargs...) : [random_walk(mean[i], sigma=sigma[i]; kwargs...) for i =eachindex(mean)];
end

@doc raw"""
    gen_series!(dt::T where T<:AbstractVector{Float64},τ::Float;N::Int64=10000, sigma::Float64=1.0)
    gen_series!(dt::T where T<:AbstractVector{Float64},τ::Vector{Float64},λ::Vector{Float64};N::Int64=1000,sigma::Float64=1.0)

Generate a Markov Chain of autocorrelated data usign the autocorrelation function `Γ(t) = e^{-t/τ}` or `Γ(t) = ∑_{k} λ_k e^{-t/τ_k}` and mean value `0` and *adds* it to `dt`

`N` is the length of the Markov Chain
`sigma` is the standar deviation of the normally distributed random number that are generated.

"""
function gen_series!(dt::T where T<:AbstractVector, τ; N = 1000,sigma=1.0)
    ap = τ == 0.0 ? 0.0 : exp(-1.0/τ)
    η= randn(N);
    dt[1] = η[1];
    for i in 2:N
        dt[i] = sqrt(1-ap^2)*η[i]*sigma+ap*dt[i-1]
    end
end

@doc raw"""
    gen_series(μ,τ;N::Int=1000)
    gen_series(μ,τ::AbstractVector,λ::AbstractVector;N::Int=1000)

Based on the homonymous Fortran90 routine written by Alberto Ramos. See https://ific.uv.es/~alramos/software/aderrors/

It generate a Markov Chain of autocorrelated data using the autocorrelation function `Γ(t) = e^{-t/τ}` or `Γ(t) = ∑_{k} λ_k e^{-t/τ_k}` and mean value `μ`

`N` is the length of the Markov Chain
`sigma` is the standar deviation of the normally distributed random number that are generated.
"""
function gen_series(mu, τ; N::Int64=1000,sigma=1.0)
    dt = zeros(N)
    gen_series!(dt,τ,N=N,sigma=sigma)
    return dt.+ mu
end

function gen_series!(dt::AbstractVector,τ::Vector,λ::Vector;N::Int64=1000,sigma=1.0)
    nu = zeros(N)
    for k in eachindex(τ)
        ap = τ[k] == 0.0 ? 0.0 : exp(-1.0/τ[k])
        η = randn(N)
        nu[1] = η[1]*sigma
        dt[1] = λ[k]*nu[1]
        for i in 2:N
            nu[i] = sqrt(1-ap[k]^2)*η[i]*sigma + ap[k]*nu[i-1]
            dt[i]+= λ[k]*nu[i]
        end
    end
end

function gen_series(μ,τ::Vector,λ::Vector;sigma::Float64=1.0, N::Int64=1000)
    dt =zeros(N);
    gen_series!(dt,τ,λ,N=N,sigma=sigma)

    return dt.+μ
end

@doc raw"""
    gen_cov_series(μ:Vector{Float64}, cov::Matrix{Float64},τ::Vector{Float64},λ::Vector{Float64};N::Int64=1000)

It generate a matrix `length(μ) × N` in which each row is the autocorrelated Montecarlo History of the corresponding μ value.

The covariance matrix computed with generated data and ignoring the autocorrelation will tend to `cov` as `N -> ∞`.
"""
function gen_cov_series(μ::Vector, cov::Matrix, τ, λ; N::Int64=1000)
    np = length(μ)
    res = zeros(np,N);
    L = LinearAlgebra.cholesky(cov).L;
    temp = similar(μ)

    for i in 1:np
        gen_series!(view(res,i,1:N),τ,λ,sigma=1.0,N=N)
    end
    for j in axes(res,2)
        LinearAlgebra.mul!(temp,L,view(res,:,j))
        res[:,j] .= temp .+μ
    end
    return res;
end

function real_taui(τ::Float64)
  if τ==0.
    return 0.5
  end
  a = exp(-1/τ)
  return 0.5 + a/(1-a)
end

function real_taui(τ::Vector{Float64},λ::Vector{Float64})
  ap = [t == 0.0 ? 0. : exp(-1.0/t) for t in τ]
  return 0.5 + sum(λ.^2 ./(1 ./ap .- 1))/sum(λ.^2)
end


export rn_gen,rn_uwreal_gen,corr_data_gen,random_walk,gen_series,gen_cov_series,gen_series!,real_taui

end
