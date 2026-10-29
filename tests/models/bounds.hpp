#pragma once

// Bounds that are not plain numbers -- the "data and parameter bounds" case.
//
//   data {
//     int<lower=1> J; vector[J] y; vector[J] L; vector[J] U;
//   }
//   parameters {
//     real mu;
//     real<lower=0> tau;
//     vector<offset=mu, multiplier=tau>[J] alpha;  // non-centred, parameter
//                                                  // offset and multiplier
//     real<lower=mu> m;                            // parameter lower bound
//     vector<lower=L>[J] x;                        // data vector bound
//     vector<lower=mu, upper=U>[J] v;              // parameter and data mixed
//   }
//
// A bound that depends on another parameter needs that parameter's value, so
// parameters() declares one statement at a time instead of in a single braced
// tuple.  Sequential statements run in order, so the read order is as safe
// as with the braced form.
//
// The model views its data and never copies it.  `L_` and `U_` are views,
// so the bound specs below hold them by reference and never copy them.

#include <stancpp/stancpp.hpp>

#include <tuple>
#include <utility>

namespace models {

struct BoundsPosterior : stancpp::model<BoundsPosterior> {
 public:
  // The model views y, L and U and never copies them; the caller keeps them
  // alive for as long as the model.  `L_` and `U_` are views, so the bound
  // specs below hold them by reference: no data is copied per evaluation.
  //
  // This model does not split into one model per group: alpha has one
  // value per group, so a subset of groups has a different parameter vector.
  template <typename DY, typename DL, typename DU>
  BoundsPosterior(DY&& y, DL&& L, DU&& U)
      : y_(stancpp::view_vector(std::forward<DY>(y), "y")),
        L_(stancpp::view_vector(std::forward<DL>(L), "L")),
        U_(stancpp::view_vector(std::forward<DU>(U), "U")),
        L_plus_1_(L_.array() + 1.0) {}

  template <typename Scope>
  auto parameters(Scope& scope) const {
    namespace spec = stancpp::spec;
    const Eigen::Index J = y_.size();

    auto mu = scope.declare("mu", spec::real{});
    auto tau = scope.declare("tau", spec::lb(spec::real{}, 0.0));
    auto alpha = scope.declare(
        "alpha", spec::offset_multiplier(spec::vector(J), mu, tau));
    auto m = scope.declare("m", spec::lb(spec::real{}, mu));
    auto x = scope.declare("x", spec::lb(spec::vector(J), L_));
    auto v = scope.declare("v", spec::lub(spec::vector(J), mu, U_));
    return std::tuple{mu, tau, alpha, m, x, v};
  }

  template <bool Propto, bool Jacobian, typename Derived>
  auto log_prob(const Eigen::MatrixBase<Derived>& theta) const {
    using T = typename Derived::Scalar;

    stancpp::reader<T, Jacobian> scope(theta);
    auto [mu, tau, alpha, m, x, v] = parameters(scope);

    T target = scope.target();
    target += stan::math::normal_lpdf<Propto>(mu, 0.0, 5.0);
    target += stan::math::lognormal_lpdf<Propto>(tau, 0.0, 1.0);
    target += stan::math::normal_lpdf<Propto>(alpha, mu, tau);
    target += stan::math::normal_lpdf<Propto>(m, mu, 2.0);
    target += stan::math::normal_lpdf<Propto>(x, L_plus_1_, 1.0);
    target += stan::math::normal_lpdf<Propto>(
        v, stan::math::divide(stan::math::add(mu, U_), 2.0), 1.0);

    target += stan::math::normal_lpdf<Propto>(y_, alpha, 1.0);

    return target;
  }

 private:
  stancpp::vector_view y_;
  stancpp::vector_view L_;
  stancpp::vector_view U_;
  // Stan's `transformed data`: `L + 1` depends only on data.
  Eigen::VectorXd L_plus_1_;
};

}  // namespace models
