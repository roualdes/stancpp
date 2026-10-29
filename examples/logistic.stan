// The Stan program that logistic.cpp writes by hand.
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
generated quantities {
  array[N] int<lower=0, upper=1> y_rep = bernoulli_logit_rng(alpha + X * beta);
  real<lower=0, upper=1> mean_y_rep = mean(y_rep);
}
