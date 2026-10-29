// The Stan program that mix.cpp writes by hand.
data {
  int<lower=1> N;
  int<lower=1> C;
  vector[N] y;
  real<lower=0> prior_weight;
}
parameters {
  simplex[C] w;
  ordered[C] mu;
  vector<lower=0>[C] sigma;
}
model {
  target += prior_weight * (normal_lpdf(mu | 0, 5)
                            + lognormal_lpdf(sigma | 0, 1)
                            + dirichlet_lpdf(w | rep_vector(2.0, C)));
  for (n in 1 : N) {
    vector[C] lps;
    for (c in 1 : C) {
      lps[c] = log(w[c]) + normal_lpdf(y[n] | mu[c], sigma[c]);
    }
    target += log_sum_exp(lps);
  }
}
