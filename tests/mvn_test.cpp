// What only the mvn model shows: a generated quantity that the log density
// also uses, through a helper both call.

#include <stancpp/stancpp.hpp>

#include <gtest/gtest.h>

#include <random>

#include "model_cases.hpp"

namespace {

// The generated quantity L is the Cholesky factor of Sigma: lower
// triangular, with L L' = Sigma.
TEST(Mvn, GeneratedLIsCholeskyFactorOfSigma) {
  const test::mvn_case c;
  std::mt19937 rng(1);
  const Eigen::VectorXd out = c.model.param_constrain(c.theta, true, rng);

  // The output starts B (P x K), Sigma (K x K), tau, then L (K x K), each
  // column major.
  const Eigen::Index P = c.X.cols();
  const Eigen::Index K = c.Yt.rows();
  const Eigen::Map<const Eigen::MatrixXd> Sigma(out.data() + P * K, K, K);
  const Eigen::Map<const Eigen::MatrixXd> L(out.data() + P * K + K * K + 1, K,
                                            K);
  EXPECT_EQ(Eigen::MatrixXd(L.triangularView<Eigen::StrictlyUpper>()),
            Eigen::MatrixXd::Zero(K, K));
  EXPECT_LE((L * L.transpose() - Sigma).cwiseAbs().maxCoeff(), 1e-12);
}

}  // namespace
