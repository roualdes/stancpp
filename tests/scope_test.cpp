// The scopes: what each does with a declaration, apart from any model.

#include <stancpp/stancpp.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace stancpp::spec;

// ---- reader ---------------------------------------------------------

// A draws matrix with theta as its second row: column major, as R stores
// it, so the row is strided; and row major, as a C-order numpy array is, so
// the row is contiguous.
TEST(Reader, ViewsContiguousThetaAndCopiesStridedTheta) {
  const Eigen::VectorXd theta = Eigen::VectorXd::LinSpaced(4, -1.0, 1.0);
  Eigen::MatrixXd draws = Eigen::MatrixXd::Zero(2, theta.size());
  draws.row(1) = theta.transpose();
  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      draws_rm = draws;

  EXPECT_FALSE(stancpp::reader<double>(theta).copied());
  EXPECT_FALSE(stancpp::reader<double>(draws_rm.row(1)).copied());
  EXPECT_TRUE(stancpp::reader<double>(draws.row(1)).copied());
}

// stan::math::gradient and stan::math::hessian build theta as a fresh
// contiguous column of autodiff values, so a reader must never copy it on
// the paths that evaluate derivatives.
TEST(Reader, NeverCopiesThetaWhenDifferentiating) {
  const Eigen::VectorXd theta = Eigen::VectorXd::LinSpaced(4, -1.0, 1.0);
  bool copied_in_gradient = true;
  bool copied_in_hessian = true;
  const auto probe = [](bool& copied) {
    return [&copied](const auto& x) {
      using S = typename std::decay_t<decltype(x)>::Scalar;
      copied = stancpp::reader<S>(x).copied();
      return stan::math::sum(x);
    };
  };
  double fx = 0;
  Eigen::VectorXd grad;
  Eigen::MatrixXd hess;
  stan::math::gradient(probe(copied_in_gradient), theta, fx, grad);
  stan::math::hessian(probe(copied_in_hessian), theta, fx, grad, hess);
  EXPECT_FALSE(copied_in_gradient);
  EXPECT_FALSE(copied_in_hessian);
}

TEST(Reader, AddsTheJacobianOnlyWhenAsked) {
  const Eigen::VectorXd theta = Eigen::VectorXd::Constant(1, 0.3);
  stancpp::reader<double, true> with(theta);
  stancpp::reader<double, false> without(theta);
  EXPECT_EQ(with.declare("x", lb(real{}, 0.0)), std::exp(0.3));
  EXPECT_EQ(without.declare("x", lb(real{}, 0.0)), std::exp(0.3));
  EXPECT_EQ(with.target(), 0.3);
  EXPECT_EQ(without.target(), 0.0);
}

TEST(Reader, RefusesThetaOfTheWrongSize) {
  const Eigen::VectorXd theta = Eigen::VectorXd::Zero(3);
  stancpp::reader<double> short_scope(theta);
  EXPECT_THROW(short_scope.declare("x", vector(4)), std::invalid_argument);
  stancpp::reader<double> long_scope(theta);
  long_scope.declare("x", vector(2));
  EXPECT_THROW(long_scope.target(), std::invalid_argument);
}

// ---- writer ---------------------------------------------------------

TEST(Writer, RejectsAQuantityThatViolatesItsBound) {
  const Eigen::VectorXd none(0);
  stancpp::writer w(none);
  EXPECT_THROW(w.emit("x", lb(real{}, 0.0), [] { return -1.0; }),
               std::domain_error);
}

TEST(Writer, RejectsAQuantityOfTheWrongSize) {
  const Eigen::VectorXd none(0);
  stancpp::writer w(none);
  EXPECT_THROW(
      w.emit("x", vector(3), [] { return Eigen::VectorXd::Zero(2).eval(); }),
      std::invalid_argument);
}

TEST(Writer, WritesIntegersAsDoubles) {
  const Eigen::VectorXd none(0);
  stancpp::writer w(none);
  w.emit("y", array(3, integer{}), [] { return std::vector<int>{2, 0, 5}; });
  EXPECT_EQ(w.generated_quantities(), Eigen::Vector3d(2.0, 0.0, 5.0));
}

// ---- schema ---------------------------------------------------------

TEST(Schema, RecordsNamesWithoutComputingTheValue) {
  stancpp::schema s;
  bool computed = false;
  s.emit("g", array(2, integer{}), [&] {
    computed = true;
    return std::vector<int>{1, 1};
  });
  EXPECT_FALSE(computed);
  EXPECT_EQ(s.generated_quantities_names(),
            (std::vector<std::string>{"g.1", "g.2"}));
  EXPECT_TRUE(s.output_names(false).empty());
}

TEST(Schema, RecordsSizesAndNamesOfBothLayouts) {
  stancpp::schema s;
  s.declare("w", simplex(3));
  s.declare("m", matrix(2, 2));
  EXPECT_EQ(s.unconstrained_size(), 2 + 4);
  EXPECT_EQ(s.constrained_size(), 3 + 4);
  EXPECT_EQ(s.constrained_names(),
            (std::vector<std::string>{"w.1", "w.2", "w.3", "m.1.1", "m.2.1",
                                      "m.1.2", "m.2.2"}));
  EXPECT_EQ(static_cast<Eigen::Index>(s.unconstrained_names().size()),
            s.unconstrained_size());
}

// ---- sizes ----------------------------------------------------------

TEST(Sizes, CountsWhatTheSchemaRecordsWithoutComputing) {
  stancpp::schema s;
  stancpp::internal::sizes n;
  bool computed = false;
  const auto quantity = [&] {
    computed = true;
    return std::vector<int>{1, 1};
  };
  s.declare("w", simplex(3));
  s.declare("m", matrix(2, 2));
  s.emit("g", array(2, integer{}), quantity);
  n.declare("w", simplex(3));
  n.declare("m", matrix(2, 2));
  n.emit("g", array(2, integer{}), quantity);
  EXPECT_FALSE(computed);
  EXPECT_EQ(n.unconstrained_size(), s.unconstrained_size());
  EXPECT_EQ(n.output_size(false), s.output_size(false));
  EXPECT_EQ(n.output_size(true), s.output_size(true));
}

// ---- unconstrainer --------------------------------------------------

TEST(Unconstrainer, InvertsTheReader) {
  const Eigen::VectorXd theta = Eigen::Vector3d(0.4, -0.7, 1.1);
  stancpp::reader<double> r(theta);
  const Eigen::VectorXd w = r.declare("w", simplex(3));
  const double s = r.declare("s", lb(real{}, 0.0));

  const std::map<std::string, Eigen::VectorXd> values = {
      {"w", w}, {"s", Eigen::VectorXd::Constant(1, s)}};
  stancpp::unconstrainer u(values);
  u.declare("w", simplex(3));
  u.declare("s", lb(real{}, 0.0));
  EXPECT_LE((u.unconstrained() - theta).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(Unconstrainer, RefusesAMissingOrMisshapenValue) {
  const std::map<std::string, Eigen::VectorXd> values = {
      {"w", Eigen::VectorXd::Constant(2, 0.5)}};
  stancpp::unconstrainer u(values);
  EXPECT_THROW(u.declare("v", vector(2)), std::invalid_argument);
  EXPECT_THROW(u.declare("w", simplex(3)), std::invalid_argument);
}

}  // namespace
