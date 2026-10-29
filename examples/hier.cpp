// Hierarchical group effects -- the "array" case.
//
//   parameters {
//     array[G] vector[K] z;             // two layouts, see below
//     cholesky_factor_corr[K] L;        // K x K used, (K choose 2) stored
//     vector<lower=0>[K] tau;
//   }
//
// This is the model that catches the array trap.  On the *unconstrained*
// scale Stan stores z[1] in full, then z[2], ...; in a CSV / cmdstanr draws
// matrix the columns run z.1.1, z.2.1, ..., z.G.1, z.1.2, ... -- the array
// index runs fastest, the opposite order.  stancpp::spec::array handles both.

#include <stancpp/stancpp.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>

#include "read_data.hpp"

namespace examples {

struct HierPosterior : stancpp::model<HierPosterior> {
 public:
  // The model views Zt and never copies it; the caller keeps it alive for as
  // long as the model.  Zt is K x G, one group per *column*, so that each
  // group is contiguous -- the memory of a C-order numpy (G, K) array, or
  // one t() of an R G x K matrix.
  //
  // This model does not split into one model per group: z has one vector
  // per group, so a subset of groups has a different parameter vector.
  template <typename DZ>
  HierPosterior(DZ&& Zt)
      : Zt_(stancpp::view_matrix(std::forward<DZ>(Zt), "Zt")),
        zero_(Eigen::VectorXd::Zero(Zt_.rows())) {}

  template <typename Scope>
  auto parameters(Scope& scope) const {
    const Eigen::Index G = Zt_.cols();
    const Eigen::Index K = Zt_.rows();
    return std::tuple{
        scope.declare("z", stancpp::spec::array(G, stancpp::spec::vector(K))),
        scope.declare("L", stancpp::spec::cholesky_factor_corr(K)),
        scope.declare("tau", stancpp::spec::lb(stancpp::spec::vector(K), 0.0))};
  }

  template <bool Propto, bool Jacobian, typename Derived>
  auto log_prob(const Eigen::MatrixBase<Derived>& theta) const {
    using T = typename Derived::Scalar;
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    stancpp::reader<T, Jacobian> scope(theta);
    auto [z, L, tau] = parameters(scope);

    T target = scope.target();
    target += stan::math::lognormal_lpdf<Propto>(tau, 0.0, 1.0);
    target += stan::math::lkj_corr_cholesky_lpdf<Propto>(L, 2.0);

    const MatrixT LSigma = stan::math::diag_pre_multiply(tau, L);
    for (Eigen::Index g = 0; g < Zt_.cols(); ++g) {
      const auto i = static_cast<std::size_t>(g);
      target += stan::math::multi_normal_cholesky_lpdf<Propto>(z[i], zero_, LSigma);
      target += stan::math::normal_lpdf<Propto>(Zt_.col(g), z[i], 0.5);
    }

    return target;
  }

 private:
  stancpp::matrix_view Zt_;
  // Stan's `transformed data`: the zero mean depends only on K.
  Eigen::VectorXd zero_;
};

}  // namespace examples

// The parameter names, and the log density's gradient and Hessian.
int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : EXAMPLES_DIR "/hier.json";
  const nlohmann::json data = examples::read_data(path);
  // One group per column.
  const Eigen::MatrixXd Zt = examples::to_matrix(data["Zobs"]).transpose();

  const examples::HierPosterior model(Zt);
  const Eigen::VectorXd theta =
      Eigen::VectorXd::Constant(model.param_unc_num(), 0.1);

  std::cout << "parameters";
  for (const auto& name : model.param_names()) {
    std::cout << ' ' << name;
  }
  std::cout << '\n';

  const auto [lp, grad, hess] = model.log_density_hessian(theta);
  std::cout << "log density " << lp << "\ngradient norm " << grad.norm()
            << "\nHessian is " << hess.rows() << " x " << hess.cols() << '\n';
  return 0;
}
