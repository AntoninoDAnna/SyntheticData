module SyntheticData
import ADerrors

import LinearAlgebra


@doc raw"""
    rn_gen(μ::T, σ::T; N::Int64=1000,data::Bool=false) where T<:Real

It generates `N` random number normally distributed around `μ` with standard deviation `σ`
It returns mean and variance of the random numbers. 
If `data` is set to `true`, it return also the random numbers
"""
function rn_gen(μ::T,σ::T;N::Int64=1000, data::Bool=false) where T<:Real
  aux = rand(d,N).* σ .+μ;
  mean =  sum(aux)/N;
  var = sum((aux.-mean).^2)/(N-1);
  return data ? (mean,var,aux) : (mean,var)
end

@doc raw"""
    rn_uwreal_gen(μ::T,σ::T;N::Int64=1000, id::String="generated data",mchist::Bool=false)::ADerrors.uwreal where T<:Float64

It generates `N` random number normally distributed around `μ` with standard deviation `σ`
It returns an `ADerrors.uwreal` with those data
If `mchist` is false, the resulting `uwreal` is made with the mean and standard deviation of the random numbers
If `mchist` is true, the random numbers are use as Montecarlo history of the resulting `uwreal`
`id` is the id used to generated the `uwreal`
"""
function rn_uwreal_gen(μ::T,σ::T;N::Int64=1000, id::String="generated data",mchist::Bool=false) where T<:Float64
  mean, var,hist = rn_gen(μ,σ;N=N);
  return mchist ? ADerrors.uwreal(hist,id) : ADerrors.uwreal([mean,sqrt(var)],id)
end

@doc raw"""
    corr_data_gen(μ::Vector,corr::Matrix,id::String;N::Int64=1000,σ = 1.0)
  
generates correlated data with mean value `μ` and correlation matrix `corr`. 

It first generates a `(mean, var)` for each `μ` (See also [`rn_gen`](@ref)) the uses `corr` and `var`s to generate a covariance matrix
Finally, it returns a vector of `ADerrors.uwreal` of `cobs`. 

`id` is used to set the label of the `uwreal` 
`σ` can be passed as parameter to change the standard deviation of normal distribution
"""
function corr_data_gen(
  μ::Vector, 
  corr::Matrix,
  id::String;
  N::Int64=1000,
  σ = 1.0) 
  
  n = length(μ); # number of data to generate;
  if size(corr) !=(n,n)
    error("dimension mismatch: the correlation matrix and the true_value vector have different dimensions")
  end
  
  if length(σ)==1
    σ = fill(σ,n);
  end

  means,vars =let aux = rn_gen.(μ,σ,N=N);
    getfield.(aux,1), getfield.(aux,2)
  end
  
  cov = [corr[i,j]*sqrt(vars[i]*vars[j]) for i = 1:n, j=1:n]

  return ADerrors.cobs(means,cov,id)    
end


@doc raw"""
    random_walk(mean::Float64; tol::Float64 = 1.0, N::Int64=1000, id::String="generated data")

it generates a random walk around

"""
function random_walk(mean::Float64; tol::Float64=1.0, N::Int64=1000, id::String="generated data") 
  x = zeros(N);
  eps = randn(N).*tol; 
  x[1] = mean + eps[1]
  for i in 2:N
    x[i] = x[i-1] + eps[i]
    if abs(x[i] - mean) > tol
      x[i] = x[i-1]
    end
  end
  return ADerrors.uwreal(x,id)
end

function random_walk(mean::Vector{Float64}; tol::Union{Vector{Float64}, Float64}=1.0, kwargs...)
  return length(tol) == 1 ? random_walk.(mean, tol=tol; kwargs...) : [random_walk(mean[i], tol=tol[i]; kwargs...) for i =eachindex(mean)];
end  


@doc raw"""
    gen_series!(dt::T where T<:AbstractVector{Float64},τ::Float;N::Int64=10000, sigma::Float64=1.0)
    gen_series!(dt::T where T<:AbstractVector{Float64},τ::Vector{Float64},λ::Vector{Float64};N::Int64=1000,sigma::Float64=1.0)

Generate a Markov Chain of autocorrelated data usign the autocorrelation function `Γ(t) = e^{-t/τ}` or `Γ(t) = ∑_{k} λ_k e^{-t/τ_k}` and mean value `0` and *adds* it to `dt`

`N` is the length of the Markov Chain
`sigma` is the standar deviation of the normally distributed random number that are generated.

"""
function gen_series!(dt::T where T<:AbstractVector{Float64}, τ::Float64;N::Int64 = 1000,sigma::Float64=1.0)
  ap = τ == 0.0 ? 0.0 : exp(-1.0/τ)
  η = randn(N).*sigma;
  dt[1] = η[1];
  for i in 1:N
    dt[i] = sqrt(1-ap^2)*η[i]+ap*dt[i-1]
  end
end

@doc raw"""
    gen_series(μ::Float64,τ::Float64;N::Int64=1000)
  
    gen_series(μ::Float64, τ::Vector{Float64},λ::Vector{Float64};N::Int64=1000,info::Bool=false)
  
based on the homonymous Fortran90 routine written by Alberto Ramos. See https://ific.uv.es/~alramos/software/aderrors/

generate a Markov Chain of autocorrelated data usign the autocorrelation function `Γ(t) = e^{-t/τ}` or `Γ(t) = ∑_{k} λ_k e^{-t/τ_k}` and mean value `μ`

`N` is the length of the Markov Chain
The flag `info`, if set to true, will output `τ_{int}` and the exact error 
`sigma` is the standar deviation of the normally distributed random number that are generated.
"""
function gen_series(μ::Float64, τ::Float64; N::Int64=1000,sigma::Float64=1.0, info::Bool=false)
  dt = zeros(N)
  gen_series!(dt,τ,N=N,sigma=sigma)

  if info 
    taui = real_taui(τ)
    err = sqrt(2*taui/N);
    return μ.+dt, taui, err
  end
    return μ.+dt;
end

function gen_series!(dt::T where T<:AbstractVector{Float64},τ::Vector{Float64},λ::Vector{Float64};N::Int64=1000,sigma::Float64=1.0)
  nu = zeros(N)
  ap = [t == 0.0 ? 0.0 : exp(-1.0/t) for t in τ]
  
  for k in eachindex(τ)
    η = rand(Distributions.Normal(0.0,sigma),N)
    nu[1] = η[1]
    nu[2:end] = [sqrt(1-ap[k]^2)*η[i] + ap[k]*nu[i-1] for i =2:N] 
    [dt[i]+= λ[k]*nu[i] for i in 1:N];
  end
end

function gen_series(μ::Float64,τ::Vector{Float64},λ::Vector{Float64};sigma::Float64=1.0, N::Int64=1000, info::Bool=false)
  dt =zeros(N);
  gen_series!(dt,τ,λ,N=N,sigma=sigma)
  dt = dt.+μ

  if info
    sλ = sum(λ.^2)
    taui = 0.5 + sum(λ.^2 ./(1 ./ap .- 1))/sλ
    err = sqrt(sλ/N * 2*taui)
    return dt, taui,err
  end
  return dt
end



@doc raw"""
    gen_cov_series(μ:Vector{Float64}, cov::Matrix{Float64},τ::Vector{Float64},λ::Vector{Float64};N::Int64=1000)

It generate a matrix `length(μ) × N` in which each row is the autocorrelated Montecarlo History of the corresponding μ value. 

To generate the MonteCarlo Chain, it first rotate the μ vector into `Y = U^{-1}*μ`, where `U`it the upper triangular matrix obtained by
a Cholesky decomposition, then a Montecarlo chain is generated for each `Y`. Finally, the Montecarlo chain is rotated back with U. 
"""
function gen_cov_series(μ::Vector{Float64}, cov::Matrix{Float64}, τ::Vector{Float64}, λ::Vector{Float64}; N::Int64=1000)
  np = length(μ)
  res = zeros(np,N); #mc history of each point
  U = LinearAlgebra.cholesky(cov).U;
  Uinv = LinearAlgebra.pinv(U);
  y = Uinv*μ

  res = zeros(np, N)
  for i in 1:np
    gen_series!(res[i,:],τ,λ,sigma=1.0,N=N)

  end
  for t in 1:N
    res[:,t] = U*res[:,t];
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

export rn_gen,rn_uwreal_gen,random_walk,gen_series,gen_cov_series,gen_series!,real_taui

end