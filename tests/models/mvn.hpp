#pragma once

// Multivariate normal regression -- the "covariance matrix" case.
//
//   data {
//     int<lower=1> N; int<lower=1> P; int<lower=1> K;
//     matrix[N, P] X; array[N] vector[K] Y;
//   }
//   parameters {
//     matrix[P, K] B;        // shaped, but unconstrained
//     cov_matrix[K] Sigma;   // K + (K choose 2) values used as K x K
//     real<lower=0> tau;     // constrained
//   }
//   generated quantities {
//     matrix[K, K] L;                  // also used by the density
//     vector[N] log_lik;               // computed
//     real<lower=0> tau_sq;            // checked against its bound
//     array[N] vector[K] Y_rep;        // drawn
//     int<lower=0, upper=1> tau_big;   // an integer
//   }
//
// The parameter block is declared once in `parameters()`; `log_prob()` is the
// only place that knows what the model *means*.

#include <stancpp/stancpp.hpp>

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace models {

struct MvnPosterior : stancpp::model<MvnPosterior> {
 public:
  // `prior_weight` multiplies the prior, and only the prior: 1 for the
  // posterior, 0 for the log likelihood alone.  N models of one observation
  // each, with weight 1/N, have log densities that sum to the posterior's.
  //
  // The model views its data and never copies it, so the caller keeps X and
  // Yt alive for as long as the model:
  //
  //   X   N x P
  //   Yt  K x N, one observation per *column*, so that each observation is
  //       contiguous.  A C-order numpy array of shape (N, K) already has
  //       exactly this memory; an R N x K matrix needs one t().
  template <typename DX, typename DY>
  MvnPosterior(DX&& X, DY&& Yt, double prior_weight)
      : X_(stancpp::view_matrix(std::forward<DX>(X), "X")),
        Yt_(stancpp::view_matrix(std::forward<DY>(Yt), "Yt")),
        wishart_scale_(Eigen::MatrixXd::Identity(Yt_.rows(), Yt_.rows())),
        prior_weight_(prior_weight) {
    if (X_.rows() != Yt_.cols()) {
      throw std::invalid_argument(
          "MvnPosterior: X has " + std::to_string(X_.rows()) +
          " rows but Yt has " + std::to_string(Yt_.cols()) +
          " columns; Yt holds one observation per column");
    }
  }

  // ---- the parameter block ----------------------------------------
  //
  // Called with stancpp::reader to get values, with stancpp::schema to get
  // sizes and names, with stancpp::writer to report constrained values, and
  // with stancpp::unconstrainer to go the other way.
  template <typename Scope>
  auto parameters(Scope& scope) const {
    const Eigen::Index P = X_.cols();
    const Eigen::Index K = Yt_.rows();
    return std::tuple{scope.declare("B", stancpp::spec::matrix(P, K)),
                      scope.declare("Sigma", stancpp::spec::cov_matrix(K)),
                      scope.declare("tau", stancpp::spec::lb(stancpp::spec::real{}, 0.0))};
  }

  // ---- the Cholesky factor of Sigma -------------------------------
  //
  // What Stan would declare as a transformed parameter: log_prob() needs it,
  // and generated_quantities() reports it.  Both call this helper, so it is
  // defined once.
  template <typename S>
  static auto cholesky_factor(const S& Sigma) {
    return stan::math::cholesky_decompose(Sigma);
  }

  // ---- generated quantities ---------------------------------------
  //
  // Stan's `generated quantities` block: computed from the parameters with
  // doubles only, and free to draw from rng.  Each value comes from a
  // lambda, so no value is computed and nothing is drawn when only names
  // are wanted.  A later quantity uses an earlier one's returned value.
  template <typename Scope, typename Parameters, typename RNG>
  auto generated_quantities(Scope& scope, const Parameters& p,
                            RNG& rng) const {
    const auto& [B, Sigma, tau] = p;
    const Eigen::Index N = X_.rows();
    const Eigen::Index K = Yt_.rows();
    auto L = scope.emit("L", stancpp::spec::matrix(K, K),
                        [&] { return cholesky_factor(Sigma); });
    const auto mean = [&](Eigen::Index n) {
      return Eigen::VectorXd((X_.row(n) * B).transpose());
    };
    auto log_lik = scope.emit("log_lik", stancpp::spec::vector(N), [&] {
      Eigen::VectorXd ll(N);
      for (Eigen::Index n = 0; n < N; ++n) {
        ll(n) = stan::math::multi_normal_cholesky_lpdf(Yt_.col(n), mean(n), L);
      }
      return ll;
    });
    auto tau_sq = scope.emit("tau_sq",
                             stancpp::spec::lb(stancpp::spec::real{}, 0.0),
                             [&] { return tau * tau; });
    auto Y_rep = scope.emit(
        "Y_rep", stancpp::spec::array(N, stancpp::spec::vector(K)), [&] {
          std::vector<Eigen::VectorXd> y(N);
          for (Eigen::Index n = 0; n < N; ++n) {
            y[n] = stan::math::multi_normal_cholesky_rng(mean(n), L, rng);
          }
          return y;
        });
    auto tau_big = scope.emit(
        "tau_big", stancpp::spec::lub(stancpp::spec::integer{}, 0, 1),
        [&] { return tau > 1 ? 1 : 0; });
    return std::tuple{L, log_lik, tau_sq, Y_rep, tau_big};
  }

  // ---- the log density --------------------------------------------
  template <bool Propto, bool Jacobian, typename Derived>
  auto log_prob(const Eigen::MatrixBase<Derived>& theta) const {
    using T = typename Derived::Scalar;
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    stancpp::reader<T, Jacobian> scope(theta);
    auto [B, Sigma, tau] = parameters(scope);

    // The prior, weighted by prior_weight; the Jacobian adjustment is left
    // to the jacobian flag.
    T prior = stan::math::lognormal_lpdf<Propto>(tau, 0.0, 1.0);
    prior += stan::math::normal_lpdf<Propto>(stan::math::to_vector(B), 0.0, tau);
    prior += stan::math::inv_wishart_lpdf<Propto>(
        Sigma, static_cast<double>(Yt_.rows() + 2), wishart_scale_);
    T target = scope.target() + prior_weight_ * prior;

    // Likelihood.  Each observation is a contiguous column of Yt, passed as
    // is; only the means, which depend on the parameters, are built here.
    const MatrixT L = cholesky_factor(Sigma);
    const MatrixT mu = stan::math::multiply(X_, B);
    for (Eigen::Index n = 0; n < X_.rows(); ++n) {
      target += stan::math::multi_normal_cholesky_lpdf<Propto>(
          Yt_.col(n), VectorT(mu.row(n).transpose()), L);
    }

    return target;
  }

 private:
  stancpp::matrix_view X_;
  stancpp::matrix_view Yt_;
  // Stan's `transformed data`: depends only on data.
  Eigen::MatrixXd wishart_scale_;
  double prior_weight_;
};

}  // namespace models
