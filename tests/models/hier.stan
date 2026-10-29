// Reference implementation of tests/models/hier.hpp.
data {
  int<lower=1> G;
  int<lower=1> K;
  array[G] vector[K] Zobs;
}
parameters {
  array[G] vector[K] z;
  cholesky_factor_corr[K] L;
  vector<lower=0>[K] tau;
}
model {
  matrix[K, K] LSigma = diag_pre_multiply(tau, L);
  target += lognormal_lpdf(tau | 0, 1);
  target += lkj_corr_cholesky_lpdf(L | 2);
  for (g in 1 : G) {
    target += multi_normal_cholesky_lpdf(z[g] | rep_vector(0, K), LSigma);
    target += normal_lpdf(Zobs[g] | z[g], 0.5);
  }
}
