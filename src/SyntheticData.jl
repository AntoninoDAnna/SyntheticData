module SyntheticData
import LinearAlgebra

@doc raw"""
    gen_data(mu, sigma; N=1000,data=false)

It draws `N` random number from the Normal Distribution \mathcal{N}(x,`mu`, `sigma`^2). It returns the observed mean and variance. If `data` is set to `true`, it returns the drawn data.
"""
function gen_data(mu,sigma;N=1000, data=false)
    aux = randn(typeof(mu),N)
    for i in eachindex(aux)
        aux[i] = aux[i]*sigma+mu;
    end
    mean =  sum(aux)/N;
    var = sum((aux[i]-mean)^2 for i in eachindex(aux))/(N-1);
    return data ? (mean,var,aux) : (mean,var)
end

@doc raw"""
    gen_cov_data(mu, cov;N=1000, data=false)

 It draws `N` vectors of length `length(mu)` from the  Multivariate Normal Distribution \mathcal{N}(x,`mu`,`cov`). It returns the observed mean and covariance matrix. If `data` is set to `true`, it returns the drawn data.
"""
function gen_cov_data(mu, cov; data=false, N::Int=1000)
    if size(cov,1)!=length(mu)
        error("Dimension mismatch: the `mu` and `cov` have uncompatible dimensions")
    end
    L = LinearAlgebra.cholesky(cov).L;
    aux = randn(eltype(mu),length(mu),N)
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
    random_walk(mean; tol = 1.0, N=1000)

  It generates a random walk around `mean` with tolerance `tol`. The random numbers are drawn from the Normal Distribution \mathcal{N}(x,`mean`, `tol`^2)$. It returns a vector with the generated data.
"""
function random_walk(mean; tol=1.0, N=1000)
    x = zeros(N);
    T = eltype(mean)
    x[1] = mean + randn(T)*tol
    for i in 2:N
        x[i] = x[i-1] + randn(T)*tol
        if abs(x[i] - mean) > tol
            x[i] = x[i-1]
        end
    end
    return x
end

__get_lambda(lambda) = k-> lambda
__get_lambda(lambda::AbstractArray) = k-> lambda[k]

@doc raw"""
    gen_series!(dt,tau [,lambda]; N=1000, sigma=1.0)

 It generates a Monte Carlo series of length `N`, with autocorrelations defined by `tau` and `lambda`. The resulting series is stored in `dt`. If `lambda` is not given, then `lambda` = 1/\sqrt{length(tau)}` is assumed.
"""
function gen_series!(dt,tau,lambda;N=1000,sigma=1.0)
    T = eltype(dt)
    nu = zeros(T,N)
    dt .= zero(T)
    _lambda = __get_lambda(lambda)
    for k in eachindex(tau)
        ap = tau[k] == 0.0 ? 0.0 : exp(-1.0/tau[k])
        nu[1] = randn(T)*sigma
        dt[1] += _lambda(k)*nu[1]
        for i in 2:N
            nu[i] = sqrt(1-ap^2)*randn(T)*sigma + ap*nu[i-1]
            dt[i]+= _lambda(k)*nu[i]
        end
    end
end

gen_series!(dt, tau; N=1000, sigma=1.0) =
    gen_series!(dt, tau,
                fill( one( eltype( tau)) / sqrt( length( tau)), length( tau)),
                N = N, sigma=sigma)

@doc raw"""
    gen_series(mu, tau[, lambda]; N=1000, sigma)

It generates a Monte Carlo Series of length `N` with autocorrelations defined by `tau` and `lambda`. If `lambda` is not given, then `lambda = 1/\sqrt{length(tau)}` is assumed. It returns the Monte Carlo series generated
"""
function gen_series(mu, tau; N::Int64=1000,sigma=1.0)
    dt = zeros(N)
    gen_series!(dt,tau,N=N,sigma=sigma)
    return dt.+ mu
end

function gen_series(μ,tau::Vector,λ::Vector;sigma::Float64=1.0, N::Int64=1000)
    dt =zeros(N);
    gen_series!(dt,tau,λ,N=N,sigma=sigma)

    return dt.+μ
end

@doc raw"""
    gen_cov_series(mu, cov, tau[, lambda]; N=1000)

t generates `length(mu)` Monte Carlo series of length `N` with autocorrelations defined by `tau` and `lambda` and covariance defined by `cov`. If `lambda` is not given, then `lambda = 1/\sqrt{length(tau)}` is assumed. It returns the Monte Carlo series generated.
"""
function gen_cov_series(mu, cov, tau, lambda; N=1000)
    T = eltype(mu)
    np = length(mu)
    res = zeros(T,np,N);
    L = LinearAlgebra.cholesky(cov).L;
    temp = similar(mu)

    for i in 1:np
        gen_series!(view(res,i,:),tau,lambda,sigma=1.0,N=N)
    end
    for j in axes(res,2)
        LinearAlgebra.mul!(temp,L,view(res,:,j))
        res[:,j] .= temp .+mu
    end
    return res;
end

gen_cov_series(mu,cov,tau;N =1000) =
    gen_cov_series(mu,cov,tau,
                   fill(one(eltype(tau))/sqrt(length(tau)),length(tau)),
                   N=N)
@doc raw"""
     real_taui(tau [,lambda])

It returns the true `\tau_{\rm int}` associate to the autocorrelation function defined by `tau` and `lambda`. If `lambda` is not given, then `lambda = 1/\sqrt{length(tau)}` is assumed.
"""
function real_taui(tau, lambda)
    aux = zero(eltype(tau))
    s = zero(eltype(lambda))
    for (t,l) in zip(tau,lambda)
        ap = t == zero(eltype(tau)) ? zero(eltype(tau)) : exp(-1.0/t)
        aux +=  l^2 / (1.0 / ap -1)
        s += l^2
    end
    return 0.5 + aux/s
end

real_taui(tau) = real_taui(tau,fill(one(eltype(tau)),lenght(tau)))

@doc raw"""
     to_cov(corr,sigma)

It returns the covariance matrix given the correlation matrix `corr` and the standard deviations `sigma`
"""
function to_cov(corr,sigma)
    cov = similar(corr)
    for C in CartesianIndices(corr)
        i,j = Tuple(C)
        cov[C] = corr[C]*sqrt(sigma[i]*sigma[j])
    end
    return cov
end

@doc raw"""
     to_corr(cov)

It returns the correlation matrix associated to the covariance matrix `cov`
"""
function to_corr(cov)
    corr = similar(cov)
    for C in CartesianIndices(cov)
        i,j = Tuple(C)
        corr[C] = cov[C]/sqrt(cov[i,i]*cov[j,j])
    end
    return corr
end


export gen_data,gen_cov_data,random_walk,gen_series!,gen_series,gen_cov_series,real_taui, to_cov, to_corr

end
