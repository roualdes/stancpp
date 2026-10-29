// A logistic regression, whose parameters are all unconstrained, so the
// parameter block adds nothing numerically.
//
// Written by hand, the parameter block would be
//
//     const T alpha = theta(0);
//     const MatrixXT beta = theta.tail(X_.cols());
//
// That works only because every parameter here is an unconstrained scalar or
// vector; it has nowhere to put a Jacobian, no notion of a declaration
// consuming fewer values than it yields, and no way to report what the
// parameters are called.
//
// The model also has generated quantities: replicated outcomes drawn from
// the posterior predictive distribution, and their mean.

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

  // Stan's `generated quantities`: computed with doubles, after the
  // parameters, and free to draw from rng.  Each value comes from a lambda,
  // so nothing is computed or drawn when only names and sizes are wanted.
  template <typename Scope, typename Parameters, typename RNG>
  auto generated_quantities(Scope& scope, const Parameters& p,
                            RNG& rng) const {
    const auto& [alpha, beta] = p;
    const Eigen::Index N = X_.rows();
    auto y_rep = scope.emit(
        "y_rep",
        stancpp::spec::array(
            N, stancpp::spec::lub(stancpp::spec::integer{}, 0, 1)),
        [&] {
          const Eigen::VectorXd eta = (X_ * beta).array() + alpha;
          return stan::math::bernoulli_logit_rng(eta, rng);
        });
    auto mean_y_rep = scope.emit(
        "mean_y_rep", stancpp::spec::lub(stancpp::spec::real{}, 0.0, 1.0),
        [&] {
          // Stan's mean() of integers is a real, but Stan Math's would
          // divide as integers, so divide here.
          return stan::math::sum(y_rep) / static_cast<double>(N);
        });
    return std::tuple{y_rep, mean_y_rep};
  }

 private:
  stancpp::matrix_view X_;
  stancpp::int_vector_view y_;
  double prior_weight_;
};

}  // namespace examples

// The log density and gradient, and each observation's log likelihood from
// one model per observation.
int main(int argc, char** argv) {
  const std::string path =
      argc > 1 ? argv[1] : EXAMPLES_DIR "/logistic.json";
  const nlohmann::json data = examples::read_data(path);
  const Eigen::MatrixXd X = examples::to_matrix(data["X"]);
  const Eigen::VectorXi y = examples::to_vector<int>(data["y"]);
  const auto prior_weight = data["prior_weight"].get<double>();

  // The model views X and y, which must outlive it.
  const examples::LogisticPosterior model(X, y, prior_weight);
  const Eigen::VectorXd theta =
      Eigen::VectorXd::Constant(model.param_unc_num(), 0.1);

  const auto [lp, grad] = model.log_density_gradient(theta);
  std::cout << "log density " << lp << "\ngradient " << grad.transpose()
            << '\n';

  // Each observation's log likelihood, constants included: a model of one
  // observation, viewing the same data, with no prior and no Jacobian.
  std::cout << "log likelihood of each observation";
  for (Eigen::Index n = 0; n < X.rows(); ++n) {
    const examples::LogisticPosterior one(X.middleRows(n, 1), y.segment(n, 1),
                                          0.0);
    std::cout << ' ' << one.log_density(theta, false, false);
  }
  std::cout << '\n';

  // One output row with the generated quantities, as in a Stan CSV.
  std::mt19937 rng(1);
  const Eigen::VectorXd row = model.param_constrain(theta, true, rng);
  const auto names = model.param_names(true);
  for (Eigen::Index i = 0; i < row.size(); ++i) {
    std::cout << names[i] << ' ' << row(i) << '\n';
  }
  return 0;
}
