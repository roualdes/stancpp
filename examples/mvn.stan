// The Stan program that mvn.cpp writes by hand.
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
}
