// Reference implementation of tests/models/mvn.hpp.
//
// Every term uses the explicit `target += ..._lpdf(...)` form so that stanc
// emits the non-propto overloads.  That makes CmdStan's `log_prob` method
// (which runs with propto = true) report the *full* log density, constants
// included, so it can be compared to the hand-written C++ exactly rather
// than up to an additive constant.
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> K;
  matrix[N, P] X;
  array[N] vector[K] Y;
  real<lower=0> prior_weight;
}
parameters {
  matrix[P, K] B;
  cov_matrix[K] Sigma;
  real<lower=0> tau;
}
model {
  matrix[K, K] L = cholesky_decompose(Sigma);
  target += prior_weight * (lognormal_lpdf(tau | 0, 1)
                            + normal_lpdf(to_vector(B) | 0, tau)
                            + inv_wishart_lpdf(Sigma | K + 2, identity_matrix(K)));
  for (n in 1 : N) {
    target += multi_normal_cholesky_lpdf(Y[n] | (X[n] * B)', L);
  }
}
generated quantities {
  matrix[K, K] L = cholesky_decompose(Sigma);
  vector[N] log_lik;
  for (n in 1 : N) {
    log_lik[n] = multi_normal_cholesky_lpdf(Y[n] | (X[n] * B)', L);
  }
  real<lower=0> tau_sq = square(tau);
  array[N] vector[K] Y_rep;
  for (n in 1 : N) {
    Y_rep[n] = multi_normal_cholesky_rng((X[n] * B)', L);
  }
  int<lower=0, upper=1> tau_big = tau > 1;
}
