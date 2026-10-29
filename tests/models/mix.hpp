#pragma once

// Gaussian mixture -- the "simplex and ordered" case.
//
//   parameters {
//     simplex[C] w;                // C - 1 unconstrained values, C used
//     ordered[C] mu;               // same count, but a constrained ordering
//     vector<lower=0>[C] sigma;    // elementwise bound
//   }

#include <stancpp/stancpp.hpp>

#include <tuple>
#include <utility>

namespace models {

struct MixPosterior : stancpp::model<MixPosterior> {
 public:
  // The model views y and never copies it; the caller keeps y alive for as
  // long as the model.
  //
  // `prior_weight` multiplies the prior, and only the prior: 1 for the
  // posterior, 0 for the log likelihood alone.
  template <typename DY>
  MixPosterior(DY&& y, Eigen::Index C, double prior_weight)
      : y_(stancpp::view_vector(std::forward<DY>(y), "y")),
        C_(C),
        dirichlet_alpha_(Eigen::VectorXd::Constant(C, 2.0)),
        prior_weight_(prior_weight) {}

  template <typename Scope>
  auto parameters(Scope& scope) const {
    return std::tuple{
        scope.declare("w", stancpp::spec::simplex(C_)), scope.declare("mu", stancpp::spec::ordered(C_)),
        scope.declare("sigma", stancpp::spec::lb(stancpp::spec::vector(C_), 0.0))};
  }

  template <bool Propto, bool Jacobian, typename Derived>
  auto log_prob(const Eigen::MatrixBase<Derived>& theta) const {
    using T = typename Derived::Scalar;
    using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    stancpp::reader<T, Jacobian> scope(theta);
    auto [w, mu, sigma] = parameters(scope);

    T prior = stan::math::normal_lpdf<Propto>(mu, 0.0, 5.0);
    prior += stan::math::lognormal_lpdf<Propto>(sigma, 0.0, 1.0);
    prior += stan::math::dirichlet_lpdf<Propto>(w, dirichlet_alpha_);
    T target = scope.target() + prior_weight_ * prior;

    const VectorT log_w = stan::math::log(w);
    VectorT lps(C_);
    for (Eigen::Index n = 0; n < y_.size(); ++n) {
      for (Eigen::Index c = 0; c < C_; ++c) {
        lps(c) = log_w(c) + stan::math::normal_lpdf<Propto>(y_(n), mu(c), sigma(c));
      }
      target += stan::math::log_sum_exp(lps);
    }

    return target;
  }

 private:
  stancpp::vector_view y_;
  Eigen::Index C_;
  // Stan's `transformed data`: depends only on data.
  Eigen::VectorXd dirichlet_alpha_;
  double prior_weight_;
};

}  // namespace models
