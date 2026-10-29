// Reference implementation of tests/models/bounds.hpp.
data {
  int<lower=1> J;
  vector[J] y;
  vector[J] L;
  vector[J] U;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector<offset=mu, multiplier=tau>[J] alpha;
  real<lower=mu> m;
  vector<lower=L>[J] x;
  vector<lower=mu, upper=U>[J] v;
}
model {
  target += normal_lpdf(mu | 0, 5);
  target += lognormal_lpdf(tau | 0, 1);
  target += normal_lpdf(alpha | mu, tau);
  target += normal_lpdf(m | mu, 2);
  target += normal_lpdf(x | L + 1, 1);
  target += normal_lpdf(v | (mu + U) / 2, 1);
  target += normal_lpdf(y | alpha, 1);
}
