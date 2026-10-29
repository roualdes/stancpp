#include <stancpp/stancpp.hpp>

#include <iostream>
#include <string>
#include <tuple>
#include <utility>

namespace examples {

struct BernoulliPosterior : stancpp::model<BernoulliPosterior> {
 public:
  template <typename DY>
  explicit BernoulliPosterior(DY&& y)
      : y_(stancpp::view_vector(std::forward<DY>(y), "y")) {}

  template <typename Scope>
  auto parameters(Scope& scope) const {
    return std::tuple{scope.declare(
        "theta", stancpp::spec::lub(stancpp::spec::real{}, 0.0, 1.0))};
  }

  template <bool Propto, bool Jacobian, typename Derived>
  auto log_prob(const Eigen::MatrixBase<Derived>& theta_unc) const {
    using T = typename Derived::Scalar;

    stancpp::reader<T, Jacobian> scope(theta_unc);
    auto [theta] = parameters(scope);

    T target = scope.target();
    target += stan::math::beta_lpdf<Propto>(theta, 1.0, 1.0);
    target += stan::math::bernoulli_lpmf<Propto>(y_, theta);
    return target;
  }

 private:
  stancpp::int_vector_view y_;
};

}  // namespace examples


int main() {
  Eigen::VectorXi y10(10);
  Eigen::VectorXi y20(20);

  const examples::BernoulliPosterior ten(y10);
  const examples::BernoulliPosterior twenty(y20);

  Eigen::VectorXd theta_unc(1);
  theta_unc << stan::math::logit(0.3);

  const auto report = [&](const std::string& name,
                          const examples::BernoulliPosterior& model) {
    const auto [lp, grad] = model.log_density_gradient(theta_unc);
    std::cout << name << ": theta "
              << model.param_constrain(theta_unc).transpose()
              << ", log density " << lp << ", gradient " << grad.transpose()
              << '\n';
  };

  report("10 flips", ten);
  report("20 flips", twenty);
  return 0;
}
