// Every spec in stancpp/spec.hpp, checked against its own definition.
//
// The model tests compare against values from CmdStan, which pin whatever
// Stan Math version CmdStan shipped with.  That is the right test for read
// order, sizes and names, but it cannot check a transform whose definition
// changed between Stan releases (the simplex did, in Stan Math 5.x).
//
// So these tests check each spec against its own definition instead:
//
//   1. constrain and unconstrain are inverses, both ways round;
//   2. the accumulated log|J| equals a numerically differentiated log
//      determinant on the transform's natural chart; and
//   3. the constrained value actually satisfies its constraint, and the
//      spec's own check, which Stan applies to generated quantities,
//      accepts it.
//
// Then each check must reject a value that violates its constraint.
//
// (2) is the one that matters: it is what makes the density on the
// unconstrained scale the right density, and it is checked here without
// trusting any particular Stan version.

#include <stancpp/stancpp.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

// Expect `ok`, describing the check when it fails.
void expect(const std::string& what, bool ok,
            const std::string& detail = "") {
  EXPECT_TRUE(ok) << what << (detail.empty() ? "" : "  (" + detail + ")");
}

// Whether `spec`'s check accepts `value`.
template <typename Spec, typename V>
bool accepts(const Spec& spec, const V& value) {
  try {
    spec.check("spec_test", "x", value);
  } catch (const std::domain_error&) {
    return false;
  }
  return true;
}

template <typename Spec>
Eigen::VectorXd flat_constrain(const Spec& spec, const Eigen::VectorXd& y) {
  double lp = 0;
  auto value = spec.template constrain<false>(y, lp);
  std::vector<double> out;
  stancpp::internal::flatten_value(value, out);
  return Eigen::Map<const Eigen::VectorXd>(
      out.data(), static_cast<Eigen::Index>(out.size()));
}

template <typename Spec>
double analytic_log_jacobian(const Spec& spec, const Eigen::VectorXd& y) {
  double lp = 0;
  spec.template constrain<true>(y, lp);
  return lp;
}

// log |det d(chart)/dy| by central differences.  `chart` selects the
// coordinates of the flattened constrained value that form the transform's
// natural chart -- for a cov_matrix, the lower triangle; for a simplex, all
// but the last component.
template <typename Spec>
double numeric_log_jacobian(const Spec& spec, const Eigen::VectorXd& y,
                            const std::vector<Eigen::Index>& chart) {
  const Eigen::Index m = y.size();
  const double h = 1e-5;
  Eigen::MatrixXd J(chart.size(), m);
  for (Eigen::Index j = 0; j < m; ++j) {
    Eigen::VectorXd hi = y;
    Eigen::VectorXd lo = y;
    hi(j) += h;
    lo(j) -= h;
    const Eigen::VectorXd chi = flat_constrain(spec, hi);
    const Eigen::VectorXd clo = flat_constrain(spec, lo);
    for (std::size_t i = 0; i < chart.size(); ++i) {
      J(static_cast<Eigen::Index>(i), j) =
          (chi(chart[i]) - clo(chart[i])) / (2 * h);
    }
  }
  return std::log(std::abs(J.determinant()));
}

std::vector<Eigen::Index> all_indices(Eigen::Index n) {
  std::vector<Eigen::Index> out;
  for (Eigen::Index i = 0; i < n; ++i) {
    out.push_back(i);
  }
  return out;
}

// Lower triangle of a K x K matrix flattened column major.
std::vector<Eigen::Index> lower_triangle(Eigen::Index rows, Eigen::Index cols,
                                         bool strict) {
  std::vector<Eigen::Index> out;
  for (Eigen::Index j = 0; j < cols; ++j) {
    for (Eigen::Index i = (strict ? j + 1 : j); i < rows; ++i) {
      out.push_back(i + rows * j);
    }
  }
  return out;
}

using StructuralCheck = std::function<bool(const Eigen::VectorXd&)>;

struct Options {
  // unit_vector is not injective: it normalises, so `unconstrain` cannot
  // recover the unconstrained value, only its direction.  Stan compensates with an
  // auxiliary -0.5 * ||y||^2 term rather than a change-of-variables Jacobian.
  bool injective = true;

  // Difference between the log determinant on the chart this test
  // differentiates and the reference measure Stan's Jacobian is defined
  // against.  Nonzero only for sum_to_zero, whose transform is an isometry
  // onto the sum-zero hyperplane: Stan reports log|J| = 0, while projecting
  // onto the first K-1 coordinates contributes a constant -0.5 * log(K).
  double chart_log_scale = 0.0;
};

template <typename Spec>
void check_spec(const std::string& name, const Spec& spec,
                const Eigen::VectorXd& y,
                const std::vector<Eigen::Index>& chart,
                const StructuralCheck& structural = nullptr,
                const Options& options = Options{}) {
  SCOPED_TRACE(name);

  if (y.size() != spec.unconstrained_size()) {
    expect("test point size", false, "test is malformed");
    return;
  }

  const Eigen::VectorXd c = flat_constrain(spec, y);
  expect("constrained_size", c.size() == spec.constrained_size(),
         std::to_string(c.size()) + " vs " +
             std::to_string(spec.constrained_size()));

  // suffix count must match the flattened layout, or CSV columns misalign
  std::vector<std::string> suffixes;
  spec.suffixes(suffixes);
  expect("suffix count",
         static_cast<Eigen::Index>(suffixes.size()) == spec.constrained_size());

  // the unconstrainer rebuilds values from this flat layout, so rebuilding
  // and flattening again must be the identity
  {
    auto value = spec.template zero<double>();
    stancpp::internal::unflatten_value(c, value);
    std::vector<double> again;
    stancpp::internal::flatten_value(value, again);
    const Eigen::VectorXd a = Eigen::Map<const Eigen::VectorXd>(
        again.data(), static_cast<Eigen::Index>(again.size()));
    expect("flatten(unflatten(c)) == c",
           a.size() == c.size() && (a - c).cwiseAbs().maxCoeff() == 0.0);
  }

  // unconstrain . constrain == identity
  const Eigen::VectorXd back = spec.unconstrain(c);
  if (options.injective) {
    const double round_err = (back - y).cwiseAbs().maxCoeff();
    expect("unconstrain(constrain(y)) == y", round_err < 1e-10,
           "max abs err " + std::to_string(round_err));
  } else {
    // The transform is not injective, so an unconstrained value cannot be
    // recovered from a constrained one.
  }

  // constrain . unconstrain == identity (on the constrained side)
  const Eigen::VectorXd c2 = flat_constrain(spec, back);
  const double round_err2 = (c2 - c).cwiseAbs().maxCoeff();
  expect("constrain(unconstrain(c)) == c", round_err2 < 1e-10,
         "max abs err " + std::to_string(round_err2));

  if (!chart.empty()) {
    const double analytic = analytic_log_jacobian(spec, y);
    const double numeric = numeric_log_jacobian(spec, y, chart);
    const double expected = analytic + options.chart_log_scale;
    const double err = std::abs(expected - numeric);
    expect("log|J| matches numerical log|det|",
           err < 1e-5 * (1 + std::abs(numeric)),
           "analytic " + std::to_string(analytic) + " vs numeric " +
               std::to_string(numeric));
  }

  if (structural) {
    expect("constrained value satisfies its constraint", structural(c));
  }

  {
    double lp = 0;
    const auto value = spec.template constrain<false>(y, lp);
    expect("check accepts the constrained value", accepts(spec, value));
  }
}

Eigen::VectorXd seq(std::initializer_list<double> v) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(v.size()));
  Eigen::Index i = 0;
  for (double x : v) {
    out(i++) = x;
  }
  return out;
}

Eigen::MatrixXd as_rect(const Eigen::VectorXd& c, Eigen::Index rows,
                        Eigen::Index cols) {
  Eigen::MatrixXd m(rows, cols);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < cols; ++j) {
    for (Eigen::Index i = 0; i < rows; ++i) {
      m(i, j) = c(k++);
    }
  }
  return m;
}

Eigen::VectorXd as_flat(const Eigen::MatrixXd& m) {
  return Eigen::Map<const Eigen::VectorXd>(m.data(), m.size());
}

Eigen::MatrixXd as_square(const Eigen::VectorXd& c, Eigen::Index n) {
  Eigen::MatrixXd m(n, n);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < n; ++j) {
    for (Eigen::Index i = 0; i < n; ++i) {
      m(i, j) = c(k++);
    }
  }
  return m;
}

const Eigen::VectorXd y1 = seq({0.37});
const Eigen::VectorXd y2 = seq({0.37, -0.82});
const Eigen::VectorXd y3 = seq({0.37, -0.82, 1.14});
const Eigen::VectorXd y6 = seq({0.37, -0.82, 1.14, -0.29, 0.65, -1.03});
const Eigen::VectorXd L3 = seq({-1.0, 0.0, 2.0});
const Eigen::VectorXd U3 = seq({1.0, 2.5, 4.0});

using namespace stancpp::spec;

TEST(Spec, UnconstrainedShapes) {
  check_spec("real", real{}, y1, all_indices(1));
  check_spec("vector[3]", vector(3), y3, all_indices(3));
  check_spec("row_vector[3]", row_vector(3), y3, all_indices(3));
  check_spec("matrix[3,2]", matrix(3, 2), y6, all_indices(6));
}

TEST(Spec, Bounds) {
  check_spec("real<lower=0>", lb(real{}, 0.0), y1, all_indices(1),
             [](const Eigen::VectorXd& c) { return c(0) > 0; });
  check_spec("vector<lower=-1>[3]", lb(vector(3), -1.0), y3, all_indices(3),
             [](const Eigen::VectorXd& c) { return (c.array() > -1).all(); });
  check_spec("real<upper=2>", ub(real{}, 2.0), y1, all_indices(1),
             [](const Eigen::VectorXd& c) { return c(0) < 2; });
  check_spec("vector<lower=0,upper=1>[3]", lub(vector(3), 0.0, 1.0), y3,
             all_indices(3), [](const Eigen::VectorXd& c) {
               return (c.array() > 0).all() && (c.array() < 1).all();
             });
  check_spec("real<offset=2,multiplier=3>", offset_multiplier(real{}, 2.0, 3.0),
             y1, all_indices(1));
  check_spec("matrix<lower=0>[3,2]", lb(matrix(3, 2), 0.0), y6, all_indices(6),
             [](const Eigen::VectorXd& c) { return (c.array() > 0).all(); });
}

TEST(Spec, BoundsGivenAsData) {
  check_spec("vector<lower=L>[3]", lb(vector(3), L3), y3, all_indices(3),
             [&](const Eigen::VectorXd& c) { return (c.array() > L3.array()).all(); });
  check_spec("vector<upper=U>[3]", ub(vector(3), U3), y3, all_indices(3),
             [&](const Eigen::VectorXd& c) { return (c.array() < U3.array()).all(); });
  // a number at one end and data at the other
  check_spec("vector<lower=0,upper=U>[3]", lub(vector(3), 0.0, U3), y3,
             all_indices(3), [&](const Eigen::VectorXd& c) {
               return (c.array() > 0).all() && (c.array() < U3.array()).all();
             });
  check_spec("vector<lower=L,upper=U>[3]", lub(vector(3), L3, U3), y3,
             all_indices(3), [&](const Eigen::VectorXd& c) {
               return (c.array() > L3.array()).all() &&
                      (c.array() < U3.array()).all();
             });
  check_spec("vector<offset=m,multiplier=s>[3]",
             offset_multiplier(vector(3), L3, seq({0.5, 2.0, 3.0})), y3,
             all_indices(3));
  // A matrix bound has to line up with the column-major flat layout; give
  // every element a different bound so a transposition would be caught.
  {
    Eigen::MatrixXd Lm(3, 2);
    Lm << -3, 1, -2, 2, -1, 3;
    const Eigen::VectorXd Lflat = as_flat(Lm);
    check_spec("matrix<lower=Lm>[3,2]", lb(matrix(3, 2), Lm), y6,
               all_indices(6), [Lflat](const Eigen::VectorXd& c) {
                 return (c.array() > Lflat.array()).all();
               });
  }
  // Data bounds, including views of data, are held by reference and never
  // copied; everything that might not outlive the declaration is stored by
  // value.
  {
    Eigen::VectorXd Lmut = L3;
    const Eigen::Map<const Eigen::VectorXd> Lmap(L3.data(), 3);
    static_assert(std::is_same_v<decltype(lb(vector(3), L3).bound.lb),
                                 const Eigen::VectorXd&>);
    static_assert(std::is_same_v<decltype(lb(vector(3), Lmut).bound.lb),
                                 const Eigen::VectorXd&>);
    static_assert(std::is_same_v<decltype(lb(vector(3), 2.0 * L3).bound.lb),
                                 Eigen::VectorXd>);
    static_assert(std::is_same_v<
                  decltype(lb(vector(3), Eigen::VectorXd(L3)).bound.lb),
                  Eigen::VectorXd>);
    static_assert(std::is_same_v<decltype(lb(vector(3), Lmap).bound.lb),
                                 const Eigen::Map<const Eigen::VectorXd>&>);
    static_assert(
        std::is_same_v<decltype(lb(real{}, 0.0).bound.lb), double>);
    static_assert(std::is_same_v<
                  decltype(lub(vector(3), 0.0, U3).bound.ub),
                  const Eigen::VectorXd&>);
    expect("data bound held by reference",
           &lb(vector(3), L3).bound.lb == &L3);
    expect("data view bound held by reference",
           lb(vector(3), Lmap).bound.lb.data() == L3.data());
  }
  // An Eigen expression as a bound is stored by value, not by reference.
  check_spec("vector<lower=2*L>[3]", lb(vector(3), 2.0 * L3), y3,
             all_indices(3), [&](const Eigen::VectorXd& c) {
               return (c.array() > 2.0 * L3.array()).all();
             });
}

TEST(Spec, ConstrainedVectors) {
  // A simplex lives on the first K-1 coordinates; the last is determined.
  check_spec("simplex[4]", simplex(4), y3, all_indices(3),
             [](const Eigen::VectorXd& c) {
               return std::abs(c.sum() - 1) < 1e-12 && (c.array() > 0).all();
             });
  check_spec("ordered[3]", ordered(3), y3, all_indices(3),
             [](const Eigen::VectorXd& c) {
               return c(0) < c(1) && c(1) < c(2);
             });
  check_spec("positive_ordered[3]", positive_ordered(3), y3, all_indices(3),
             [](const Eigen::VectorXd& c) {
               return c(0) > 0 && c(0) < c(1) && c(1) < c(2);
             });
  // Stan's sum_to_zero transform is an isometry onto the sum-zero hyperplane,
  // so it reports log|J| = 0; differentiating the first K-1 coordinates
  // instead picks up a constant -0.5 * log(K), which cannot affect inference.
  check_spec("sum_to_zero[4]", sum_to_zero(4), y3, all_indices(3),
             [](const Eigen::VectorXd& c) {
               return std::abs(c.sum()) < 1e-12;
             },
             Options{true, -0.5 * std::log(4.0)});
  // unit_vector's "Jacobian" is an auxiliary-variable normalisation, not a
  // change-of-variables determinant, so only the structural check applies.
  check_spec("unit_vector[3]", unit_vector(3), y3, {},
             [](const Eigen::VectorXd& c) {
               return std::abs(c.norm() - 1) < 1e-12;
             },
             Options{false, 0.0});
}

TEST(Spec, ConstrainedMatrices) {
  check_spec("cov_matrix[3]", cov_matrix(3), y6, lower_triangle(3, 3, false),
             [](const Eigen::VectorXd& c) {
               const Eigen::MatrixXd m = as_square(c, 3);
               return (m - m.transpose()).cwiseAbs().maxCoeff() < 1e-12 &&
                      m.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff() >
                          0;
             });
  check_spec("corr_matrix[3]", corr_matrix(3), y3, lower_triangle(3, 3, true),
             [](const Eigen::VectorXd& c) {
               const Eigen::MatrixXd m = as_square(c, 3);
               return (m.diagonal().array() - 1).abs().maxCoeff() < 1e-12 &&
                      (m - m.transpose()).cwiseAbs().maxCoeff() < 1e-12;
             });
  check_spec("cholesky_factor_corr[3]", cholesky_factor_corr(3), y3,
             lower_triangle(3, 3, true), [](const Eigen::VectorXd& c) {
               const Eigen::MatrixXd m = as_square(c, 3);
               return (m.rowwise().norm().array() - 1).abs().maxCoeff() < 1e-12;
             });
  check_spec("cholesky_factor_cov[3,2]", cholesky_factor_cov(3, 2),
             seq({0.37, -0.82, 1.14, -0.29, 0.65}),
             lower_triangle(3, 2, false), [](const Eigen::VectorXd& c) {
               // lower trapezoidal with positive diagonal
               return c(3) == 0.0 && c(0) > 0 && c(4) > 0;
             });
}

TEST(Spec, MatrixShapedConstrainedTypes) {
  // Each column is a simplex, so the chart drops the last row of each column.
  check_spec("column_stochastic_matrix[3,2]", column_stochastic_matrix(3, 2),
             seq({0.37, -0.82, 1.14, -0.29}), {0, 1, 3, 4},
             [](const Eigen::VectorXd& c) {
               return std::abs(c.segment(0, 3).sum() - 1) < 1e-12 &&
                      std::abs(c.segment(3, 3).sum() - 1) < 1e-12 &&
                      (c.array() > 0).all();
             });
  // Each row is a simplex, so the chart drops the last column.
  check_spec("row_stochastic_matrix[2,3]", row_stochastic_matrix(2, 3),
             seq({0.37, -0.82, 1.14, -0.29}), {0, 1, 2, 3},
             [](const Eigen::VectorXd& c) {
               const Eigen::MatrixXd m = as_rect(c, 2, 3);
               return (m.rowwise().sum().array() - 1).abs().maxCoeff() < 1e-12 &&
                      (c.array() > 0).all();
             });
  // Like the vector case, an isometry: Stan reports log|J| = 0 and the
  // top-left (N-1)x(M-1) chart differs by a constant, checked below.
  check_spec("sum_to_zero_matrix[3,3]", sum_to_zero_matrix(3, 3),
             seq({0.37, -0.82, 1.14, -0.29}), {},
             [](const Eigen::VectorXd& c) {
               const Eigen::MatrixXd m = as_rect(c, 3, 3);
               return m.rowwise().sum().cwiseAbs().maxCoeff() < 1e-12 &&
                      m.colwise().sum().cwiseAbs().maxCoeff() < 1e-12;
             });
  {
    // Stan reports log|J| = 0 for sum_to_zero because the transform is an
    // isometry.  The claim that actually matters is that the chart-based
    // determinant is *constant in y* -- if it varied, a zero Jacobian would
    // be wrong rather than merely offset.  Check that directly.
    const auto mat = sum_to_zero_matrix(3, 3);
    const double m1 = numeric_log_jacobian(mat, seq({0.37, -0.82, 1.14, -0.29}),
                                           {0, 1, 3, 4});
    const double m2 = numeric_log_jacobian(mat, seq({-1.6, 0.9, 0.05, 2.2}),
                                           {0, 1, 3, 4});
    expect("sum_to_zero_matrix[3,3]", std::abs(m1 - m2) < 1e-6,
           std::to_string(m1) + " vs " + std::to_string(m2));

    const auto vec = sum_to_zero(4);
    const double v1 = numeric_log_jacobian(vec, seq({0.37, -0.82, 1.14}),
                                           all_indices(3));
    const double v2 = numeric_log_jacobian(vec, seq({-1.6, 0.9, 0.05}),
                                           all_indices(3));
    expect("sum_to_zero[4]", std::abs(v1 - v2) < 1e-6,
           std::to_string(v1) + " vs " + std::to_string(v2));
  }
}

TEST(Spec, Arrays) {
  // array[2] vector[3]: 6 unconstrained, 6 constrained, but the CSV layout
  // runs the array index fastest while the values are stored slowest-first.
  check_spec("array[2] vector[3]", array(2, vector(3)), y6, all_indices(6));
  check_spec("array[2] simplex[3]", array(2, simplex(3)), seq({0.4, -0.7, 1.1, -0.2}),
             {0, 1, 2, 3});
  check_spec("array[2] real<lower=0>", array(2, lb(real{}, 0.0)), y2,
             all_indices(2),
             [](const Eigen::VectorXd& c) { return (c.array() > 0).all(); });
}

TEST(Spec, ChecksRejectViolations) {
  {
    Eigen::MatrixXd not_symmetric = Eigen::MatrixXd::Identity(3, 3);
    not_symmetric(0, 1) = 0.5;
    Eigen::MatrixXd upper = Eigen::MatrixXd::Identity(3, 3);
    upper(0, 2) = 0.5;
    expect("real<lower=0> rejects -1", !accepts(lb(real{}, 0.0), -1.0));
    expect("real<upper=2> rejects 3", !accepts(ub(real{}, 2.0), 3.0));
    expect("vector<lower=L>[3] rejects a value below L",
           !accepts(lb(vector(3), L3), seq({0.0, 0.0, 0.0})));
    expect("simplex[3] rejects (0.5, 0.5, 0.5)",
           !accepts(simplex(3), seq({0.5, 0.5, 0.5})));
    expect("ordered[3] rejects (1, 0, 2)",
           !accepts(ordered(3), seq({1.0, 0.0, 2.0})));
    expect("cov_matrix[3] rejects a non-symmetric matrix",
           !accepts(cov_matrix(3), not_symmetric));
    expect("cholesky_factor_cov[3] rejects an upper element",
           !accepts(cholesky_factor_cov(3), upper));
    expect("int<lower=0, upper=1> rejects 2",
           !accepts(lub(integer{}, 0, 1), 2));
    expect("int<lower=0, upper=1> accepts 1",
           accepts(lub(integer{}, 0, 1), 1));
    expect("array[2] real<lower=0> rejects a negative element",
           !accepts(array(2, lb(real{}, 0.0)), std::vector<double>{1, -1}));
    expect("unbounded shapes accept anything",
           accepts(real{}, -1e300) && accepts(matrix(3, 3), not_symmetric) &&
               accepts(array(2, integer{}), std::vector<int>{-4, 9}));
  }
}

}  // namespace
