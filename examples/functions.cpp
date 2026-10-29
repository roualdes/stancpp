// A robust regression with a scale that changes with x -- the "functions"
// case.
//
//   functions {
//     real laplace_lpdf(real y, real mu, real b);   // a density
//     real softplus(real x);                        // a hand-written gradient
//   }
//   parameters {
//     real alpha; real beta;      // the mean, alpha + beta * x
//     real gamma0; real gamma1;   // the scale, softplus(gamma0 + gamma1 * x)
//   }
//
// Stan's `functions` block is ordinary C++.  A function that log_prob()
// calls is differentiated along with the rest of it, provided it is a
// template on its arguments' types: log_prob() runs with plain doubles,
// with Stan Math's reverse-mode `var` for gradients, and with `fvar<var>`
// for Hessians.

#include <stancpp/stancpp.hpp>

#include <nlohmann/json.hpp>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "read_data.hpp"

namespace examples {

// The Laplace density, written as Stan Math writes its own.  The return
// type combines the arguments' types, so data stays double and only
// parameters carry derivatives, and each term is kept only when Propto
// says to, as `normal_lpdf<Propto>` does.
template <bool Propto, typename T_y, typename T_loc, typename T_scale>
stan::return_type_t<T_y, T_loc, T_scale> laplace_lpdf(const T_y& y,
                                                      const T_loc& mu,
                                                      const T_scale& b) {
  using stan::math::include_summand;
  stan::return_type_t<T_y, T_loc, T_scale> lp = 0;
  if (include_summand<Propto, T_y, T_loc, T_scale>::value) {
    lp -= stan::math::fabs(y - mu) / b;
  }
  if (include_summand<Propto, T_scale>::value) {
    lp -= stan::math::log(b);
  }
  if (include_summand<Propto>::value) {
    lp -= stan::math::LOG_TWO;
  }
  return lp;
}

// softplus(x) = log(1 + exp(x)), for any scalar type.
template <typename T>
T softplus(const T& x) {
  return stan::math::log1p_exp(x);
}

// The same function with its derivative, inv_logit(x), written by hand.
// C++ prefers this overload for `var`, so gradients use it; the template
// above still serves doubles and the `fvar<var>` of Hessians.
inline stan::math::var softplus(const stan::math::var& x) {
  return stan::math::make_callback_var(
      stan::math::log1p_exp(x.val()), [x](auto& vi) mutable {
        x.adj() += vi.adj() * stan::math::inv_logit(x.val());
      });
}

struct LaplaceRegression : stancpp::model<LaplaceRegression> {
 public:
  // The model views x and y and never copies them; the caller keeps both
  // alive for as long as the model.
  template <typename DX, typename DY>
  LaplaceRegression(DX&& x, DY&& y)
      : x_(stancpp::view_vector(std::forward<DX>(x), "x")),
        y_(stancpp::view_vector(std::forward<DY>(y), "y")) {
    if (x_.size() != y_.size()) {
      throw std::invalid_argument("LaplaceRegression: x has " +
                                  std::to_string(x_.size()) +
                                  " values but y has " +
                                  std::to_string(y_.size()));
    }
  }

  template <typename Scope>
  auto parameters(Scope& scope) const {
    return std::tuple{scope.declare("alpha", stancpp::spec::real{}),
                      scope.declare("beta", stancpp::spec::real{}),
                      scope.declare("gamma0", stancpp::spec::real{}),
                      scope.declare("gamma1", stancpp::spec::real{})};
  }

  template <bool Propto, bool Jacobian, typename Derived>
  auto log_prob(const Eigen::MatrixBase<Derived>& theta) const {
    using T = typename Derived::Scalar;

    stancpp::reader<T, Jacobian> scope(theta);
    auto [alpha, beta, gamma0, gamma1] = parameters(scope);

    T target = scope.target();
    target += stan::math::normal_lpdf<Propto>(alpha, 0.0, 5.0);
    target += stan::math::normal_lpdf<Propto>(beta, 0.0, 5.0);
    target += stan::math::normal_lpdf<Propto>(gamma0, 0.0, 2.0);
    target += stan::math::normal_lpdf<Propto>(gamma1, 0.0, 2.0);
    for (Eigen::Index n = 0; n < y_.size(); ++n) {
      const T b = softplus(gamma0 + gamma1 * x_(n));
      target += laplace_lpdf<Propto>(y_(n), alpha + beta * x_(n), b);
    }

    return target;
  }

 private:
  stancpp::vector_view x_;
  stancpp::vector_view y_;
};

}  // namespace examples

// The log density and its gradient, checked against finite differences,
// and the gradient again from the Hessian method, which uses the template
// rather than the hand-written softplus.
int main(int argc, char** argv) {
  const std::string path =
      argc > 1 ? argv[1] : EXAMPLES_DIR "/functions.json";
  const nlohmann::json data = examples::read_data(path);
  const Eigen::VectorXd x = examples::to_vector(data["x"]);
  const Eigen::VectorXd y = examples::to_vector(data["y"]);

  const examples::LaplaceRegression model(x, y);
  Eigen::VectorXd theta(model.param_unc_num());
  theta << 0.8, 1.5, -0.3, 0.6;

  const auto [lp, grad] = model.log_density_gradient(theta, false);
  Eigen::VectorXd finite(theta.size());
  const double h = 1e-6;
  for (Eigen::Index i = 0; i < theta.size(); ++i) {
    Eigen::VectorXd up = theta;
    Eigen::VectorXd down = theta;
    up(i) += h;
    down(i) -= h;
    finite(i) =
        (model.log_density(up, false) - model.log_density(down, false)) /
        (2 * h);
  }
  const auto [lp_h, grad_h, hess] = model.log_density_hessian(theta, false);

  std::cout << std::setprecision(10) << "log density " << lp
            << "\ngradient " << grad.transpose() << "\nfinite differences "
            << finite.transpose() << "\ngradient from the Hessian method "
            << grad_h.transpose() << '\n';
  const double err = (grad - finite).cwiseAbs().maxCoeff();
  if (err > 1e-6 || (grad - grad_h).cwiseAbs().maxCoeff() > 1e-10) {
    std::cerr << "the gradients disagree\n";
    return 1;
  }
  return 0;
}
