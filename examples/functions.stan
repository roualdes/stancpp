// The Stan program that functions.cpp writes by hand.
//
// Every term uses the explicit `target += ..._lpdf(...)` form so that
// CmdStan's `log_prob` method reports the full log density, constants
// included.
functions {
  real laplace_lpdf(real y, real mu, real b) {
    return -abs(y - mu) / b - log(b) - log(2);
  }
  real softplus(real x) {
    return log1p_exp(x);
  }
}
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real gamma0;
  real gamma1;
}
model {
  target += normal_lpdf(alpha | 0, 5) + normal_lpdf(beta | 0, 5);
  target += normal_lpdf(gamma0 | 0, 2) + normal_lpdf(gamma1 | 0, 2);
  for (n in 1 : N) {
    target += laplace_lpdf(y[n] | alpha + beta * x[n],
                           softplus(gamma0 + gamma1 * x[n]));
  }
}
