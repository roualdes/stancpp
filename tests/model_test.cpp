// Every test model through the model methods: against CmdStan's reference
// values, and against itself.

#include <stancpp/stancpp.hpp>

#include <boost/random/additive_combine.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "model_cases.hpp"
#include "test_util.hpp"

namespace {

template <typename Case>
class ModelTest : public ::testing::Test {
 protected:
  Case c;
};

using Cases = ::testing::Types<test::logistic_case, test::mvn_case,
                               test::mix_case, test::hier_case,
                               test::bounds_case>;
TYPED_TEST_SUITE(ModelTest, Cases, test::case_names);

// The full log density, constants included, with the Jacobian, as CmdStan's
// log_prob method reports it for programs written with `target +=`.
TYPED_TEST(ModelTest, LogDensityMatchesCmdStan) {
  const auto& c = this->c;
  const nlohmann::json& ref = c.ref["log_prob"];
  const auto [lp, grad] = c.model.log_density_gradient(c.theta, false, true);
  test::expect_near(lp, ref["lp"].get<double>(), 1e-10);
  test::expect_near(grad, test::to_vector(ref["gradient"]), 1e-10);
}

// The output names, in Stan's CSV order, and the generated quantities of
// one CmdStan draw, computed from the draw's parameters, except values drawn
// at random. The parameters themselves only round trip here; the log
// density test checks the transforms against CmdStan.
TYPED_TEST(ModelTest, OutputMatchesCmdStanDraw) {
  const auto& c = this->c;
  const nlohmann::json& draw = c.ref["draw"];
  const auto names = c.model.param_names(true);
  ASSERT_EQ(names, draw["names"].get<std::vector<std::string>>());

  const Eigen::VectorXd values = test::to_vector(draw["values"]);
  const Eigen::VectorXd theta =
      c.model.param_unconstrain(values.head(c.model.param_num()));
  std::mt19937 rng(1);
  const Eigen::VectorXd out = c.model.param_constrain(theta, true, rng);
  ASSERT_EQ(out.size(), values.size());
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    const std::string& name = names[i];
    bool random = false;
    for (const std::string& prefix : TypeParam::random) {
      random = random || name.rfind(prefix + ".", 0) == 0;
    }
    if (!random) {
      SCOPED_TRACE(name);
      test::expect_near(out(i), values(i), 1e-8);
    }
  }
}

TYPED_TEST(ModelTest, SizesAndNamesAgree) {
  const auto& c = this->c;
  const stancpp::schema s = stancpp::internal::model_schema(c.model);
  EXPECT_EQ(c.model.param_unc_num(), c.theta.size());
  EXPECT_EQ(c.model.param_unc_num(), s.unconstrained_size());
  EXPECT_EQ(static_cast<Eigen::Index>(c.model.param_unc_names().size()),
            s.unconstrained_size());
  EXPECT_EQ(c.model.param_num(), s.constrained_size());
  EXPECT_EQ(c.model.param_names(), s.constrained_names());
  for (const bool gq : {true, false}) {
    EXPECT_EQ(c.model.param_num(gq), s.output_size(gq));
    EXPECT_EQ(c.model.param_names(gq), s.output_names(gq));
    EXPECT_EQ(static_cast<Eigen::Index>(c.model.param_names(gq).size()),
              c.model.param_num(gq));
  }
}

TYPED_TEST(ModelTest, ConstrainAndUnconstrainAreInverses) {
  const auto& c = this->c;
  const Eigen::VectorXd values = c.model.param_constrain(c.theta);
  Eigen::VectorXd values_out(c.model.param_num());
  c.model.param_constrain(c.theta, values_out);
  EXPECT_EQ(values_out, values);

  test::expect_near(c.model.param_unconstrain(values), c.theta, 1e-8);
  Eigen::VectorXd theta_out(c.model.param_unc_num());
  c.model.param_unconstrain(values, theta_out);
  test::expect_near(theta_out, c.theta, 1e-8);

  const Eigen::MatrixXd draws = values.transpose().replicate(2, 1);
  const Eigen::MatrixXd back = stancpp::unconstrain_draws(c.model, draws);
  test::expect_near(back.row(1).transpose(), c.theta, 1e-8);
}

// For every setting of the flags, the methods that return derivatives
// return the same log density, and their buffer versions write what they
// return.
TYPED_TEST(ModelTest, DensityMethodsAgree) {
  const auto& c = this->c;
  const auto& theta = c.theta;
  const Eigen::Index D = theta.size();
  const Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(D, -1.0, 1.0);
  for (const bool propto : {true, false}) {
    for (const bool jacobian : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "propto " << propto
                                        << ", jacobian " << jacobian);
      const double lp = c.model.log_density(theta, propto, jacobian);
      const auto [lp_g, g] =
          c.model.log_density_gradient(theta, propto, jacobian);
      const auto [lp_h, g_h, H] =
          c.model.log_density_hessian(theta, propto, jacobian);
      const auto [lp_v, Hv] = c.model.log_density_hessian_vector_product(
          theta, v, propto, jacobian);
      test::expect_near(lp_g, lp, 1e-12);
      test::expect_near(lp_h, lp, 1e-12);
      test::expect_near(lp_v, lp, 1e-12);
      test::expect_near(g_h, g, 1e-10);
      test::expect_near(Hv, Eigen::VectorXd(H * v), 1e-10);

      Eigen::VectorXd g_out(D);
      EXPECT_EQ(c.model.log_density_gradient(theta, g_out, propto, jacobian),
                lp_g);
      EXPECT_EQ(g_out, g);
      Eigen::VectorXd g_h_out(D);
      Eigen::MatrixXd H_out(D, D);
      EXPECT_EQ(c.model.log_density_hessian(theta, g_h_out, H_out, propto,
                                            jacobian),
                lp_h);
      EXPECT_EQ(g_h_out, g_h);
      EXPECT_EQ(H_out, H);
      Eigen::VectorXd Hv_out(D);
      EXPECT_EQ(c.model.log_density_hessian_vector_product(theta, v, Hv_out,
                                                           propto, jacobian),
                lp_v);
      EXPECT_EQ(Hv_out, Hv);
    }
  }
}

TYPED_TEST(ModelTest, HessianIsSymmetricAndFinite) {
  const auto& c = this->c;
  const auto [lp, g, H] = c.model.log_density_hessian(c.theta, false);
  ASSERT_EQ(H.rows(), c.theta.size());
  EXPECT_TRUE(H.allFinite());
  EXPECT_LE((H - H.transpose()).cwiseAbs().maxCoeff(), 1e-8);
}

// propto drops only terms that do not depend on the parameters: the log
// density moves by a constant, and its derivatives do not move at all.
TYPED_TEST(ModelTest, ProptoDropsOnlyConstants) {
  const auto& c = this->c;
  const Eigen::VectorXd theta2 = 0.5 * c.theta;
  for (const bool jacobian : {true, false}) {
    SCOPED_TRACE(::testing::Message() << "jacobian " << jacobian);
    const double d1 = c.model.log_density(c.theta, false, jacobian) -
                      c.model.log_density(c.theta, true, jacobian);
    const double d2 = c.model.log_density(theta2, false, jacobian) -
                      c.model.log_density(theta2, true, jacobian);
    EXPECT_NE(d1, 0.0) << "propto dropped nothing; does log_prob pass "
                          "Propto to its densities?";
    test::expect_near(d2, d1, 1e-9);
    const auto [lp_p, g_p, H_p] =
        c.model.log_density_hessian(c.theta, true, jacobian);
    const auto [lp_f, g_f, H_f] =
        c.model.log_density_hessian(c.theta, false, jacobian);
    test::expect_near(g_p, g_f, 1e-10);
    test::expect_near(H_p, H_f, 1e-9);
  }
}

// The Jacobian reaches the log density only if log_prob() starts from
// scope.target(). A model that starts from zero still constrains correctly
// but silently drops the Jacobian, so check the difference directly.
TYPED_TEST(ModelTest, JacobianReachesLogDensity) {
  const auto& c = this->c;
  stancpp::reader<double, true> scope(c.theta);
  c.model.parameters(scope);
  const double log_jacobian = scope.target();
  test::expect_near(c.model.log_density(c.theta, false, true) -
                        c.model.log_density(c.theta, false, false),
                    log_jacobian, 1e-9);
}

// A draws matrix with theta as its second row: column major, as R stores
// it, so the row is strided; and row major, as a C-order numpy array is, so
// the row is contiguous. Either gives the same log density.
TYPED_TEST(ModelTest, ThetaLayoutDoesNotMatter) {
  const auto& c = this->c;
  Eigen::MatrixXd draws = Eigen::MatrixXd::Zero(2, c.theta.size());
  draws.row(1) = c.theta.transpose();
  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      draws_rm = draws;
  const double lp = c.model.log_density(c.theta);
  EXPECT_EQ(c.model.log_density(draws.row(1)), lp);
  EXPECT_EQ(c.model.log_density(draws_rm.row(1)), lp);
}

// The same seed gives the same draws, whichever generator is used, and the
// parameters at the front of the row never depend on the draws.
TYPED_TEST(ModelTest, GeneratedQuantitiesFollowTheSeed) {
  const auto& c = this->c;
  const Eigen::Index P = c.model.param_num();
  if (c.model.param_num(true) == P) {
    GTEST_SKIP() << "the model has no generated quantities";
  }
  const Eigen::VectorXd values = c.model.param_constrain(c.theta);

  std::mt19937 rng1(7);
  std::mt19937 rng2(7);
  std::mt19937 rng3(8);
  const Eigen::VectorXd all = c.model.param_constrain(c.theta, true, rng1);
  Eigen::VectorXd all_out(c.model.param_num(true));
  c.model.param_constrain(c.theta, all_out, true, rng2);
  const Eigen::VectorXd other = c.model.param_constrain(c.theta, true, rng3);
  EXPECT_EQ(all.size(), c.model.param_num(true));
  EXPECT_EQ(all_out, all);
  EXPECT_EQ(all.head(P), values);
  EXPECT_EQ(other.head(P), values);

  boost::ecuyer1988 stan_rng(7);
  EXPECT_EQ(c.model.param_constrain(c.theta, false, stan_rng), values);
}

}  // namespace
