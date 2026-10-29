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
//     matrix[K, K] L = cholesky_decompose(Sigma);
//   }
//
// The parameter block is declared once in `parameters()`; `log_prob()` is the
// only place that knows what the model *means*.

#include <stancpp/stancpp.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "read_data.hpp"

namespace examples {

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

  // ---- generated quantities ---------------------------------------
  //
  // Reported after the parameters.  The value is produced by a lambda so
  // that scopes which only need the shape. -- `schema`, building names and
  // sizes with no parameter vector in hand -- never evaluate it.  That
  // matters here: cholesky_decompose of the schema's dummy zero matrix
  // would throw.
  template <typename Scope, typename Parameters, typename RNG>
  auto generated_quantities(Scope& scope, const Parameters& p, RNG&) const {
    const auto& [B, Sigma, tau] = p;
    const Eigen::Index K = Yt_.rows();
    return std::tuple{scope.emit("L", stancpp::spec::matrix(K, K),
                                 [&] { return cholesky_factor(Sigma); })};
  }

 private:
  stancpp::matrix_view X_;
  stancpp::matrix_view Yt_;
  // Stan's `transformed data`: depends only on data.
  Eigen::MatrixXd wishart_scale_;
  double prior_weight_;
};

}  // namespace examples

// The constrained parameters with the generated quantity, and each
// observation's log density, gradient and Hessian from one model per
// observation.
int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : EXAMPLES_DIR "/mvn.json";
  const nlohmann::json data = examples::read_data(path);
  const Eigen::MatrixXd X = examples::to_matrix(data["X"]);
  // One observation per column, so that each observation is contiguous.
  const Eigen::MatrixXd Yt = examples::to_matrix(data["Y"]).transpose();
  const auto prior_weight = data["prior_weight"].get<double>();

  const examples::MvnPosterior model(X, Yt, prior_weight);
  const Eigen::VectorXd theta = Eigen::VectorXd::Zero(model.param_unc_num());

  std::cout << model.param_unc_num() << " unconstrained parameters, "
            << model.param_num(true)
            << " values with the generated quantity:\n";
  std::mt19937 rng(1);
  const Eigen::VectorXd draw = model.param_constrain(theta, true, rng);
  const auto names = model.param_names(true);
  for (Eigen::Index i = 0; i < draw.size(); ++i) {
    std::cout << "  " << names[i] << " = " << draw(i) << '\n';
  }

  // A model of each observation, viewing the same data, with 1/N of the
  // prior, so that they sum to the whole.  The Jacobian is off, since each
  // model would add all of it.
  const Eigen::Index N = X.rows();
  const Eigen::Index D = theta.size();
  double lp_sum = 0.0;
  Eigen::MatrixXd hess_sum = Eigen::MatrixXd::Zero(D, D);
  for (Eigen::Index n = 0; n < N; ++n) {
    const examples::MvnPosterior one(X.middleRows(n, 1), Yt.middleCols(n, 1),
                                     prior_weight / static_cast<double>(N));
    const auto [lp, grad, hess] = one.log_density_hessian(theta, true, false);
    lp_sum += lp;
    hess_sum += hess;
  }
  const auto [lp, grad, hess] = model.log_density_hessian(theta, true, false);
  std::cout << "sum of per-observation log densities " << lp_sum
            << "\nwhole log density " << lp
            << "\nlargest difference between the summed and whole Hessians "
            << (hess_sum - hess).cwiseAbs().maxCoeff() << '\n';
  return 0;
}
