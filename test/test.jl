using Revise,SyntheticData

(mean,var) = rn_gen(1.0,0.1)mean,cov = cov_data_gen([1.2,2.4],[0.1^2 0.001; 0.001 0.1^2])
data = random_walk(1.0,sigma=0.05,N=10)
N = 1000

res = gen_series(1.0,25,N=N,sigma=0.1)

res = gen_cov_series([1.2,2.4],[0.1^2 0.001; 0.001 0.1^2],[25],[1], N=N)

mean = sum(res,dims=2)/N
C = [sum((res[i,t]-mean[i])*(res[j,t]-mean[j]) for t in 1:N) for i in (1,2), j in (1,2)]./(N-1)
