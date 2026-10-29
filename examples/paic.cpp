// The Posterior Averaging Information Criterion (Zhou, Entropy 2023,
// 25(3):468) for a logistic regression,
//
//   PAIC = -2 sum_i E[log p(y_i | theta) | y] + 2 tr(J^-1 I),
//
// where J and I are built at the posterior mode from each observation's log
// likelihood plus 1/n of the log prior:
//
//   J = -(1/n)     sum_i  Hessian_i,
//   I =  1/(n - 1) sum_i  gradient_i gradient_i'.
//
// Each piece comes from a model of one observation, built over views of the
// same data with the prior weighted as that piece needs.  The posterior
// draws for the first term come from a random-walk Metropolis sampler.

#include <stancpp/stancpp.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "read_data.hpp"

namespace examples {

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

    T target = stan::math::normal_lpdf<Propto>(alpha, 0.0, 10.0);
    target += stan::math::normal_lpdf<Propto>(beta, 0.0, 2.5);
    target *= prior_weight_;
    target += scope.target();

    target += stan::math::bernoulli_logit_glm_lpmf<Propto>(y_, X_, alpha, beta);

    return target;
  }

 private:
  stancpp::matrix_view X_;
  stancpp::int_vector_view y_;
  double prior_weight_;
};

}  // namespace examples

// The posterior mode, by Newton's method with step halving.  Every
// parameter is unconstrained, so the Jacobian stays off throughout.
Eigen::VectorXd posterior_mode(const examples::LogisticPosterior& model) {
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(model.param_unc_num());
  for (int iteration = 0; iteration < 100; ++iteration) {
    const auto [lp, grad, hess] = model.log_density_hessian(theta, true, false);
    if (grad.norm() < 1e-10) {
      return theta;
    }
    Eigen::VectorXd step = hess.ldlt().solve(grad);
    while (model.log_density(theta - step, true, false) < lp) {
      step /= 2;
    }
    theta -= step;
  }
  throw std::runtime_error("Newton's method did not converge");
}

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : EXAMPLES_DIR "/paic.json";
  const nlohmann::json data = examples::read_data(path);
  const Eigen::MatrixXd X = examples::to_matrix(data["X"]);
  const Eigen::VectorXi y = examples::to_vector<int>(data["y"]);
  const auto prior_weight = data["prior_weight"].get<double>();

  const Eigen::Index n = X.rows();
  const examples::LogisticPosterior posterior(X, y, prior_weight);
  const Eigen::Index D = posterior.param_unc_num();

  // ---- the bias correction, at the posterior mode -------------------
  const Eigen::VectorXd mode = posterior_mode(posterior);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(D, D);
  Eigen::MatrixXd I = Eigen::MatrixXd::Zero(D, D);
  for (Eigen::Index i = 0; i < n; ++i) {
    const examples::LogisticPosterior one(
        X.middleRows(i, 1), y.segment(i, 1),
        prior_weight / static_cast<double>(n));
    const auto [lp, grad, hess] = one.log_density_hessian(mode, true, false);
    J -= hess;
    I += grad * grad.transpose();
  }
  J /= static_cast<double>(n);
  I /= static_cast<double>(n - 1);
  const double bias = 2 * J.ldlt().solve(I).trace();

  // ---- the posterior mean of each observation's log likelihood ------
  // Random-walk Metropolis, with proposals shaped by the posterior's
  // curvature at the mode.
  const auto [lp_mode, grad_mode, hess_mode] =
      posterior.log_density_hessian(mode, true, false);
  const Eigen::MatrixXd L =
      Eigen::MatrixXd(-hess_mode).inverse().llt().matrixL();
  const double scale = 2.38 / std::sqrt(static_cast<double>(D));

  std::mt19937 rng(1);
  std::normal_distribution<double> normal;
  std::uniform_real_distribution<double> uniform;
  const int warmup = 2000;
  const int draws = 200000;

  Eigen::VectorXd theta = mode;
  double lp = lp_mode;
  Eigen::VectorXd mean_loglik = Eigen::VectorXd::Zero(n);
  for (int s = 0; s < warmup + draws; ++s) {
    const Eigen::VectorXd z =
        Eigen::VectorXd::NullaryExpr(D, [&] { return normal(rng); });
    const Eigen::VectorXd proposal = theta + scale * L * z;
    const double lp_proposal = posterior.log_density(proposal, true, false);
    if (std::log(uniform(rng)) < lp_proposal - lp) {
      theta = proposal;
      lp = lp_proposal;
    }
    if (s < warmup) {
      continue;
    }
    // Each observation's log likelihood, constants included: a model of
    // one observation with no prior.
    for (Eigen::Index i = 0; i < n; ++i) {
      const examples::LogisticPosterior one(X.middleRows(i, 1),
                                            y.segment(i, 1), 0.0);
      mean_loglik(i) += one.log_density(theta, false, false) / draws;
    }
  }
  const double fit = -2 * mean_loglik.sum();

  std::cout << "posterior mode " << mode.transpose()
            << "\n-2 sum of posterior mean log likelihoods " << fit
            << "\n2 tr(J^-1 I) " << bias << "\nPAIC " << fit + bias << '\n';
  return 0;
}
