// The Stan program that paic.cpp writes by hand.
data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
  int<lower=0> K;
  matrix[N, K] X;
  real<lower=0> prior_weight;
}
parameters {
  real alpha;
  vector[K] beta;
}
model {
  target += prior_weight * (normal_lpdf(alpha | 0, 10)
                            + normal_lpdf(beta | 0, 2.5));
  y ~ bernoulli_logit_glm(X, alpha, beta);
}
