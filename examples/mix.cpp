// Gaussian mixture -- the "simplex and ordered" case.
//
//   parameters {
//     simplex[C] w;                // C - 1 unconstrained values, C used
//     ordered[C] mu;               // same count, but a constrained ordering
//     vector<lower=0>[C] sigma;    // elementwise bound
//   }

#include <stancpp/stancpp.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <utility>

#include "read_data.hpp"

namespace examples {

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

}  // namespace examples

// Constrained draws, and each observation's log likelihood at each draw
// from one model per observation.
int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : EXAMPLES_DIR "/mix.json";
  const nlohmann::json data = examples::read_data(path);
  const Eigen::VectorXd y = examples::to_vector(data["y"]);
  const auto C = data["C"].get<Eigen::Index>();
  const auto prior_weight = data["prior_weight"].get<double>();

  const examples::MixPosterior model(y, C, prior_weight);

  // Three draws, one per row, on the unconstrained scale.
  std::mt19937 rng(1);
  std::normal_distribution<double> normal;
  const Eigen::MatrixXd draws = Eigen::MatrixXd::NullaryExpr(
      3, model.param_unc_num(), [&] { return normal(rng); });

  for (Eigen::Index m = 0; m < draws.rows(); ++m) {
    std::cout << "draw " << m << ": "
              << model.param_constrain(draws.row(m).transpose()).transpose()
              << '\n';
  }

  // One row per draw and one column per observation: a model of each
  // observation, viewing the same data, with no prior and no Jacobian.
  Eigen::MatrixXd loglik(draws.rows(), y.size());
  for (Eigen::Index n = 0; n < y.size(); ++n) {
    const examples::MixPosterior one(y.segment(n, 1), C, 0.0);
    for (Eigen::Index m = 0; m < draws.rows(); ++m) {
      loglik(m, n) = one.log_density(draws.row(m), false, false);
    }
  }
  std::cout << "log likelihoods are " << loglik.rows() << " x "
            << loglik.cols() << "; each row sums to its draw's log likelihood:\n"
            << loglik.rowwise().sum().transpose() << '\n';
  return 0;
}
