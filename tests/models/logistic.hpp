#pragma once

// A logistic regression, whose parameters are all unconstrained.
//
// With no constraints, the parameter block adds nothing numerically, and
// test_logistic checks the model agrees with logistic.stan on data.json.
//
// Written by hand, the parameter block would be
//
//     const T alpha = theta(0);
//     const MatrixXT beta = theta.tail(X_.cols());
//
// That works only because every parameter here is an unconstrained scalar or
// vector; it has nowhere to put a Jacobian, no notion of a declaration
// consuming fewer values than it yields, and no way to tell R what the
// parameters are called.

#include <stancpp/stancpp.hpp>

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace models {

struct LogisticPosterior : stancpp::model<LogisticPosterior> {
 public:
  // `prior_weight` multiplies the prior: 1 for the posterior, 0 for the log
  // likelihood alone.
  //
  // The model views X (N x P) and the integer outcomes y (N) and never
  // copies them; the caller keeps both alive for as long as the model.
  template <typename DX, typename DY>
  LogisticPosterior(DX&& X, DY&& y, double prior_weight)
      : X_(stancpp::view_matrix(std::forward<DX>(X), "X")),
        y_(stancpp::view_vector(std::forward<DY>(y), "y")),
        prior_weight_(prior_weight) {
    if (X_.rows() != y_.size()) {
      throw std::invalid_argument("LogisticPosterior: X has " +
                                  std::to_string(X_.rows()) +
                                  " rows but y has " +
                                  std::to_string(y_.size()) + " values");
    }
  }

  template <typename Scope>
  auto parameters(Scope& scope) const {
    return std::tuple{scope.declare("alpha", stancpp::spec::real{}),
                      scope.declare("beta", stancpp::spec::vector(X_.cols()))};
  }

  template <bool Propto, bool Jacobian, typename Derived>
  auto log_prob(const Eigen::MatrixBase<Derived>& theta) const {
    using T = typename Derived::Scalar;

    stancpp::reader<T, Jacobian> scope(theta);
    auto [alpha, beta] = parameters(scope);

    T prior = stan::math::normal_lpdf<Propto>(alpha, 0.0, 10.0);
    prior += stan::math::normal_lpdf<Propto>(beta, 0.0, 2.5);
    T target = scope.target() + prior_weight_ * prior;

    target += stan::math::bernoulli_logit_glm_lpmf<Propto>(y_, X_, alpha, beta);

    return target;
  }

 private:
  stancpp::matrix_view X_;
  stancpp::int_vector_view y_;
  double prior_weight_;
};

}  // namespace models
