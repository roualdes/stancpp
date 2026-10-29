// Models whose likelihood splits into units, one model per unit viewing the
// same data: the unit models must sum to the whole model's log density,
// gradient and Hessian, whatever weight the caller gives each unit's prior.
//
// The Jacobian is left out, since each unit model adds the whole of it.

#include <stancpp/stancpp.hpp>

#include <gtest/gtest.h>

#include "model_cases.hpp"
#include "test_util.hpp"

namespace {

template <typename Case>
class DecompositionTest : public ::testing::Test {
 protected:
  Case c;
};

using Cases =
    ::testing::Types<test::logistic_case, test::mvn_case, test::mix_case>;
TYPED_TEST_SUITE(DecompositionTest, Cases, test::case_names);

// Sum the units' log densities, gradients and Hessians, and compare them with
// the whole model's.
template <typename Case, typename Whole>
void expect_units_sum_to(const Case& c, const Whole& whole,
                         double unit_prior_weight) {
  const auto& theta = c.theta;
  const Eigen::Index D = theta.size();
  for (const bool propto : {true, false}) {
    SCOPED_TRACE(::testing::Message() << "propto " << propto);
    double lp_sum = 0.0;
    Eigen::VectorXd g_sum = Eigen::VectorXd::Zero(D);
    Eigen::MatrixXd H_sum = Eigen::MatrixXd::Zero(D, D);
    for (Eigen::Index n = 0; n < c.units(); ++n) {
      const auto [lp, g, H] = c.unit(n, unit_prior_weight)
                                  .log_density_hessian(theta, propto, false);
      lp_sum += lp;
      g_sum += g;
      H_sum += H;
    }
    const auto [lp, g, H] = whole.log_density_hessian(theta, propto, false);
    test::expect_near(lp_sum, lp, 1e-9);
    test::expect_near(g_sum, g, 1e-9);
    test::expect_near(H_sum, H, 1e-9);
  }
}

// With 1/N of the prior each, the units sum to the posterior.
TYPED_TEST(DecompositionTest, UnitsSumToPosterior) {
  const auto& c = this->c;
  expect_units_sum_to(c, c.whole(1.0), 1.0 / static_cast<double>(c.units()));
}

// With no prior, the units sum to the log likelihood.
TYPED_TEST(DecompositionTest, UnitsSumToLogLikelihood) {
  const auto& c = this->c;
  expect_units_sum_to(c, c.whole(0.0), 0.0);
}

}  // namespace
