#pragma once

// Stan Math extends Eigen's matrices, which works only when Stan Math is
// included before Eigen.
#if defined(EIGEN_CORE_H) && !defined(EIGEN_MATRIXBASE_PLUGIN)
#error "stancpp: include Stan Math, or stancpp, before any Eigen header"
#endif

#include <stan/math/mix.hpp>

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace stancpp {

// Implementation details of the specs.
namespace internal {

/**
 * @brief Return `n` choose 2.
 *
 * @param[in] n The size of a square matrix.
 * @return The number of elements strictly below its diagonal.
 */
inline Eigen::Index tri(Eigen::Index n) { return (n * (n - 1)) / 2; }

/**
 * @brief Reshape a vector into a matrix in column major order.
 *
 * @tparam Vec The type of the vector.
 * @param[in] u The values.
 * @param[in] rows The number of rows.
 * @param[in] cols The number of columns.
 * @return The matrix.
 */
template <typename Vec>
Eigen::Matrix<typename Vec::Scalar, Eigen::Dynamic, Eigen::Dynamic>
to_matrix_colmajor(const Vec& u, Eigen::Index rows, Eigen::Index cols) {
  Eigen::Matrix<typename Vec::Scalar, Eigen::Dynamic, Eigen::Dynamic> m(rows,
                                                                        cols);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < cols; ++j) {
    for (Eigen::Index i = 0; i < rows; ++i) {
      m.coeffRef(i, j) = u.coeff(k++);
    }
  }
  return m;
}

/**
 * @brief Flatten a matrix in column major order.
 *
 * @param[in] m The matrix.
 * @return The values.
 */
inline Eigen::VectorXd flatten_colmajor(const Eigen::MatrixXd& m) {
  Eigen::VectorXd out(m.size());
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < m.cols(); ++j) {
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
      out(k++) = m(i, j);
    }
  }
  return out;
}

/**
 * @brief Append the suffixes `.1` to `.n`.
 *
 * @param[in] n The number of values.
 * @param[out] out The suffixes.
 */
inline void flat_suffixes(Eigen::Index n, std::vector<std::string>& out) {
  for (Eigen::Index i = 1; i <= n; ++i) {
    out.emplace_back("." + std::to_string(i));
  }
}

/**
 * @brief Append the Stan CSV suffixes of a matrix, such as `.1.2`.
 *
 * @param[in] rows The number of rows.
 * @param[in] cols The number of columns.
 * @param[out] out The suffixes, first index fastest.
 */
inline void matrix_suffixes(Eigen::Index rows, Eigen::Index cols,
                            std::vector<std::string>& out) {
  for (Eigen::Index j = 1; j <= cols; ++j) {
    for (Eigen::Index i = 1; i <= rows; ++i) {
      out.emplace_back("." + std::to_string(i) + "." + std::to_string(j));
    }
  }
}

/**
 * @brief How a bound argument of type `B` is stored.
 *
 * A named Eigen object, such as a model's data, is held by const
 * reference and never copied. Anything else, such as a number, a
 * parameter, or an expression like `2.0 * L`, is stored by value.
 */
template <typename B>
using bound_storage_t = std::conditional_t<
    std::is_lvalue_reference<B>::value &&
        stan::is_eigen<std::decay_t<B>>::value,
    const std::decay_t<B>&, stan::plain_type_t<B>>;

/**
 * @brief Return a bound as `foo_free` takes it: a double, or an Eigen
 * bound flattened in column major order.
 *
 * @tparam B The type of the bound.
 * @param[in] b The bound.
 * @return The bound's values.
 */
template <typename B>
auto flat_bound(const B& b) {
  if constexpr (stan::is_eigen<B>::value) {
    return Eigen::VectorXd(
        stan::math::to_vector(stan::math::value_of_rec(b)));
  } else {
    return static_cast<double>(stan::math::value_of_rec(b));
  }
}

}  // namespace internal

/**
 * @brief Specs for Stan's parameter types.
 *
 * A spec describes one declaration, such as `vector<lower=0>[K]`, and
 * provides
 *  - `value_t<T>`, the type of the usable value with scalar type `T`,
 *  - `unconstrained_size()`, the number of values read from theta,
 *  - `constrained_size()`, the number of values in the flattened value,
 *  - `constrain<Jacobian>(u, lp)`, which maps unconstrained values `u` to
 *    a usable value and adds the log absolute Jacobian determinant to `lp`,
 *  - `unconstrain(c)`, the inverse, from flattened constrained values,
 *  - `suffixes(out)`, the Stan CSV name suffixes, such as `.1.2`,
 *  - `unconstrained_suffixes(out)`, name suffixes for the values read from
 *    theta, in theta's order,
 *  - `zero<T>()`, a correctly shaped value of zeros, and
 *  - `check(function, name, v)`, which throws `std::domain_error` when a
 *    computed value `v` violates the constraint, as Stan checks
 *    generated quantities.
 *
 * Flattened values are column major, the order of a Stan CSV row.
 */
namespace spec {

/**
 * @brief Spec for `real`.
 */
struct real {
  template <typename T>
  using value_t = T;

  Eigen::Index unconstrained_size() const { return 1; }
  Eigen::Index constrained_size() const { return 1; }

  template <bool Jacobian, typename Vec, typename Lp>
  typename Vec::Scalar constrain(const Vec& u, Lp&) const {
    return u.coeff(0);
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return c;
  }

  void suffixes(std::vector<std::string>& out) const { out.emplace_back(""); }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    out.emplace_back("");
  }

  template <typename T>
  T zero() const {
    return T(0);
  }

  template <typename V>
  void check(const char*, const std::string&, const V&) const {}
};

/**
 * @brief Spec for `int`, which only a generated quantity may be.
 *
 * It has no unconstrained form, so declaring it as a parameter does not
 * compile. Its values are written as doubles, as in a Stan CSV row.
 */
struct integer {
  template <typename T>
  using value_t = int;

  Eigen::Index constrained_size() const { return 1; }

  void suffixes(std::vector<std::string>& out) const { out.emplace_back(""); }

  template <typename T>
  int zero() const {
    return 0;
  }

  template <typename V>
  void check(const char*, const std::string&, const V&) const {}
};

/**
 * @brief Spec for `vector[n]`.
 */
struct vector {
  Eigen::Index n;

  /**
   * @brief Construct the spec.
   *
   * @param[in] n The number of elements.
   */
  explicit vector(Eigen::Index n) : n(n) {}

  template <typename T>
  using value_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  Eigen::Index unconstrained_size() const { return n; }
  Eigen::Index constrained_size() const { return n; }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp&) const {
    return u;
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return c;
  }

  void suffixes(std::vector<std::string>& out) const {
    for (Eigen::Index i = 1; i <= n; ++i) {
      out.emplace_back("." + std::to_string(i));
    }
  }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    suffixes(out);
  }

  template <typename T>
  value_t<T> zero() const {
    return value_t<T>::Zero(n);
  }

  template <typename V>
  void check(const char*, const std::string&, const V&) const {}
};

/**
 * @brief Spec for `row_vector[n]`.
 */
struct row_vector {
  Eigen::Index n;

  /**
   * @brief Construct the spec.
   *
   * @param[in] n The number of elements.
   */
  explicit row_vector(Eigen::Index n) : n(n) {}

  template <typename T>
  using value_t = Eigen::Matrix<T, 1, Eigen::Dynamic>;

  Eigen::Index unconstrained_size() const { return n; }
  Eigen::Index constrained_size() const { return n; }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp&) const {
    return u.transpose();
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return c;
  }

  void suffixes(std::vector<std::string>& out) const {
    for (Eigen::Index i = 1; i <= n; ++i) {
      out.emplace_back("." + std::to_string(i));
    }
  }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    suffixes(out);
  }

  template <typename T>
  value_t<T> zero() const {
    return value_t<T>::Zero(n);
  }

  template <typename V>
  void check(const char*, const std::string&, const V&) const {}
};

/**
 * @brief Spec for `matrix[rows, cols]`.
 */
struct matrix {
  Eigen::Index rows;
  Eigen::Index cols;

  /**
   * @brief Construct the spec.
   *
   * @param[in] rows The number of rows.
   * @param[in] cols The number of columns.
   */
  matrix(Eigen::Index rows, Eigen::Index cols) : rows(rows), cols(cols) {}

  template <typename T>
  using value_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  Eigen::Index unconstrained_size() const { return rows * cols; }
  Eigen::Index constrained_size() const { return rows * cols; }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp&) const {
    value_t<typename Vec::Scalar> out(rows, cols);
    Eigen::Index k = 0;
    for (Eigen::Index j = 0; j < cols; ++j) {
      for (Eigen::Index i = 0; i < rows; ++i) {
        out.coeffRef(i, j) = u.coeff(k++);
      }
    }
    return out;
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return c;
  }

  void suffixes(std::vector<std::string>& out) const {
    for (Eigen::Index j = 1; j <= cols; ++j) {
      for (Eigen::Index i = 1; i <= rows; ++i) {
        out.emplace_back("." + std::to_string(i) + "." + std::to_string(j));
      }
    }
  }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    suffixes(out);
  }

  template <typename T>
  value_t<T> zero() const {
    return value_t<T>::Zero(rows, cols);
  }

  template <typename V>
  void check(const char*, const std::string&, const V&) const {}
};

/**
 * @brief Elementwise bounds, each holding its own bound values.
 *
 * A bound may be a number, data the same shape as the value, or a
 * parameter declared earlier. Each provides `constrain<Jacobian>(x, lp)`,
 * `unconstrain(c)`, and `check(function, name, v)`.
 */
namespace bound {

/**
 * @brief A lower bound, `<lower=lb>`.
 *
 * @tparam L The type of the bound.
 */
template <typename L>
struct lower {
  L lb;

  template <bool Jacobian, typename X, typename Lp>
  auto constrain(const X& x, Lp& lp) const {
    return stan::math::lb_constrain<Jacobian>(x, lb, lp);
  }
  Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) const {
    return stan::math::lb_free(c, internal::flat_bound(lb));
  }
  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    stan::math::check_greater_or_equal(function, name.c_str(), v,
                                       stan::math::value_of_rec(lb));
  }
};

/**
 * @brief An upper bound, `<upper=ub>`.
 *
 * @tparam U The type of the bound.
 */
template <typename U>
struct upper {
  U ub;

  template <bool Jacobian, typename X, typename Lp>
  auto constrain(const X& x, Lp& lp) const {
    return stan::math::ub_constrain<Jacobian>(x, ub, lp);
  }
  Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) const {
    return stan::math::ub_free(c, internal::flat_bound(ub));
  }
  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    stan::math::check_less_or_equal(function, name.c_str(), v,
                                    stan::math::value_of_rec(ub));
  }
};

/**
 * @brief Lower and upper bounds, `<lower=lb, upper=ub>`.
 *
 * @tparam L The type of the lower bound.
 * @tparam U The type of the upper bound.
 */
template <typename L, typename U>
struct lower_upper {
  L lb;
  U ub;

  template <bool Jacobian, typename X, typename Lp>
  auto constrain(const X& x, Lp& lp) const {
    return stan::math::lub_constrain<Jacobian>(x, lb, ub, lp);
  }
  Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) const {
    return stan::math::lub_free(c, internal::flat_bound(lb),
                                internal::flat_bound(ub));
  }
  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    stan::math::check_bounded(function, name.c_str(), v,
                              stan::math::value_of_rec(lb),
                              stan::math::value_of_rec(ub));
  }
};

/**
 * @brief An affine transform, `<offset=offset, multiplier=multiplier>`.
 *
 * @tparam O The type of the offset.
 * @tparam M The type of the multiplier.
 */
template <typename O, typename M>
struct offset_multiplier {
  O offset;
  M multiplier;

  template <bool Jacobian, typename X, typename Lp>
  auto constrain(const X& x, Lp& lp) const {
    return stan::math::offset_multiplier_constrain<Jacobian>(x, offset,
                                                             multiplier, lp);
  }
  Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) const {
    return stan::math::offset_multiplier_free(
        c, internal::flat_bound(offset), internal::flat_bound(multiplier));
  }
  template <typename V>
  void check(const char*, const std::string&, const V&) const {}
};

}  // namespace bound

/**
 * @brief Spec for a shape with an elementwise bound.
 *
 * @tparam Shape The spec of the unbounded shape.
 * @tparam Bound The bound.
 */
template <typename Shape, typename Bound>
struct bounded {
  Shape shape;
  Bound bound;

  template <typename T>
  using value_t = typename Shape::template value_t<T>;

  Eigen::Index unconstrained_size() const { return shape.unconstrained_size(); }
  Eigen::Index constrained_size() const { return shape.constrained_size(); }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp& lp) const {
    return bound.template constrain<Jacobian>(
        shape.template constrain<false>(u, lp), lp);
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return bound.unconstrain(Eigen::VectorXd(c));
  }

  void suffixes(std::vector<std::string>& out) const { shape.suffixes(out); }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    shape.unconstrained_suffixes(out);
  }

  template <typename T>
  value_t<T> zero() const {
    return shape.template zero<T>();
  }

  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    shape.check(function, name, v);
    bound.check(function, name, v);
  }
};

/**
 * @brief Spec type for `Shape<lower=L>`.
 */
template <typename Shape, typename L = double>
using lb_t = bounded<Shape, bound::lower<L>>;

/**
 * @brief Spec type for `Shape<upper=U>`.
 */
template <typename Shape, typename U = double>
using ub_t = bounded<Shape, bound::upper<U>>;

/**
 * @brief Spec type for `Shape<lower=L, upper=U>`.
 */
template <typename Shape, typename L = double, typename U = double>
using lub_t = bounded<Shape, bound::lower_upper<L, U>>;

/**
 * @brief Spec type for `Shape<offset=O, multiplier=M>`.
 */
template <typename Shape, typename O = double, typename M = double>
using offset_multiplier_t = bounded<Shape, bound::offset_multiplier<O, M>>;

/**
 * @brief Return a spec for `shape` with a lower bound.
 *
 * @tparam Shape The spec of the shape.
 * @tparam L The type of the bound.
 * @param[in] shape The shape.
 * @param[in] lb The lower bound.
 * @return The spec.
 */
template <typename Shape, typename L>
lb_t<Shape, internal::bound_storage_t<L>> lb(Shape shape, L&& lb) {
  return {shape, {std::forward<L>(lb)}};
}

/**
 * @brief Return a spec for `shape` with an upper bound.
 *
 * @tparam Shape The spec of the shape.
 * @tparam U The type of the bound.
 * @param[in] shape The shape.
 * @param[in] ub The upper bound.
 * @return The spec.
 */
template <typename Shape, typename U>
ub_t<Shape, internal::bound_storage_t<U>> ub(Shape shape, U&& ub) {
  return {shape, {std::forward<U>(ub)}};
}

/**
 * @brief Return a spec for `shape` with lower and upper bounds.
 *
 * @tparam Shape The spec of the shape.
 * @tparam L The type of the lower bound.
 * @tparam U The type of the upper bound.
 * @param[in] shape The shape.
 * @param[in] lb The lower bound.
 * @param[in] ub The upper bound.
 * @return The spec.
 */
template <typename Shape, typename L, typename U>
lub_t<Shape, internal::bound_storage_t<L>, internal::bound_storage_t<U>> lub(
    Shape shape, L&& lb, U&& ub) {
  return {shape, {std::forward<L>(lb), std::forward<U>(ub)}};
}

/**
 * @brief Return a spec for `shape` with an offset and multiplier.
 *
 * @tparam Shape The spec of the shape.
 * @tparam O The type of the offset.
 * @tparam M The type of the multiplier.
 * @param[in] shape The shape.
 * @param[in] offset The offset.
 * @param[in] multiplier The multiplier.
 * @return The spec.
 */
template <typename Shape, typename O, typename M>
offset_multiplier_t<Shape, internal::bound_storage_t<O>,
                    internal::bound_storage_t<M>>
offset_multiplier(Shape shape, O&& offset, M&& multiplier) {
  return {shape, {std::forward<O>(offset), std::forward<M>(multiplier)}};
}

/**
 * @brief Stan Math's transforms, one policy each.
 *
 * Each policy gives its unconstrained size, `constrain<Jacobian>`,
 * `unconstrain`, which is Stan Math's `foo_free`, and `check`, which is
 * Stan Math's `check_foo`.
 */
namespace transform {

/**
 * @brief `simplex[n]`, from `n - 1` values.
 */
struct simplex {
  static Eigen::Index unconstrained_size(Eigen::Index n) { return n - 1; }
  template <bool Jacobian, typename Vec, typename Lp>
  static auto constrain(const Vec& u, Lp& lp) {
    return stan::math::simplex_constrain<Jacobian>(u, lp);
  }
  static Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) {
    return stan::math::simplex_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_simplex(function, name.c_str(), v);
  }
};

/**
 * @brief `ordered[n]`, from `n` values.
 */
struct ordered {
  static Eigen::Index unconstrained_size(Eigen::Index n) { return n; }
  template <bool Jacobian, typename Vec, typename Lp>
  static auto constrain(const Vec& u, Lp& lp) {
    return stan::math::ordered_constrain<Jacobian>(u, lp);
  }
  static Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) {
    return stan::math::ordered_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_ordered(function, name.c_str(), v);
  }
};

/**
 * @brief `positive_ordered[n]`, from `n` values.
 */
struct positive_ordered {
  static Eigen::Index unconstrained_size(Eigen::Index n) { return n; }
  template <bool Jacobian, typename Vec, typename Lp>
  static auto constrain(const Vec& u, Lp& lp) {
    return stan::math::positive_ordered_constrain<Jacobian>(u, lp);
  }
  static Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) {
    return stan::math::positive_ordered_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_positive_ordered(function, name.c_str(), v);
  }
};

/**
 * @brief `unit_vector[n]`, from `n` values.
 */
struct unit_vector {
  static Eigen::Index unconstrained_size(Eigen::Index n) { return n; }
  template <bool Jacobian, typename Vec, typename Lp>
  static auto constrain(const Vec& u, Lp& lp) {
    return stan::math::unit_vector_constrain<Jacobian>(u, lp);
  }
  static Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) {
    return stan::math::unit_vector_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_unit_vector(function, name.c_str(), v);
  }
};

/**
 * @brief `sum_to_zero_vector[n]`, from `n - 1` values.
 */
struct sum_to_zero {
  static Eigen::Index unconstrained_size(Eigen::Index n) { return n - 1; }
  template <bool Jacobian, typename Vec, typename Lp>
  static auto constrain(const Vec& u, Lp& lp) {
    return stan::math::sum_to_zero_constrain<Jacobian>(u, lp);
  }
  static Eigen::VectorXd unconstrain(const Eigen::VectorXd& c) {
    return stan::math::sum_to_zero_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_sum_to_zero(function, name.c_str(), v);
  }
};

/**
 * @brief `cov_matrix[n]`, from `n + tri(n)` values.
 */
struct cov_matrix {
  static Eigen::Index unconstrained_size(Eigen::Index n) {
    return n + internal::tri(n);
  }
  template <bool Jacobian, typename Vec, typename Lp>
  static auto constrain(const Vec& u, Eigen::Index n, Lp& lp) {
    return stan::math::cov_matrix_constrain<Jacobian>(u, n, lp);
  }
  static Eigen::VectorXd unconstrain(const Eigen::MatrixXd& c) {
    return stan::math::cov_matrix_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_cov_matrix(function, name.c_str(), v);
  }
};

/**
 * @brief `corr_matrix[n]`, from `tri(n)` values.
 */
struct corr_matrix {
  static Eigen::Index unconstrained_size(Eigen::Index n) {
    return internal::tri(n);
  }
  template <bool Jacobian, typename Vec, typename Lp>
  static auto constrain(const Vec& u, Eigen::Index n, Lp& lp) {
    return stan::math::corr_matrix_constrain<Jacobian>(u, n, lp);
  }
  static Eigen::VectorXd unconstrain(const Eigen::MatrixXd& c) {
    return stan::math::corr_matrix_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_corr_matrix(function, name.c_str(), v);
  }
};

/**
 * @brief `cholesky_factor_corr[n]`, from `tri(n)` values.
 */
struct cholesky_factor_corr {
  static Eigen::Index unconstrained_size(Eigen::Index n) {
    return internal::tri(n);
  }
  template <bool Jacobian, typename Vec, typename Lp>
  static auto constrain(const Vec& u, Eigen::Index n, Lp& lp) {
    return stan::math::cholesky_corr_constrain<Jacobian>(u, n, lp);
  }
  static Eigen::VectorXd unconstrain(const Eigen::MatrixXd& c) {
    return stan::math::cholesky_corr_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_cholesky_factor_corr(function, name.c_str(), v);
  }
};

/**
 * @brief `column_stochastic_matrix[n, m]`, from `(n - 1) x m` values.
 */
struct column_stochastic {
  static Eigen::Index unconstrained_rows(Eigen::Index rows) { return rows - 1; }
  static Eigen::Index unconstrained_cols(Eigen::Index cols) { return cols; }
  template <bool Jacobian, typename Mat, typename Lp>
  static auto constrain(const Mat& u, Lp& lp) {
    return stan::math::stochastic_column_constrain<Jacobian>(u, lp);
  }
  static Eigen::MatrixXd unconstrain(const Eigen::MatrixXd& c) {
    return stan::math::stochastic_column_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_stochastic_column(function, name.c_str(), v);
  }
};

/**
 * @brief `row_stochastic_matrix[n, m]`, from `n x (m - 1)` values.
 */
struct row_stochastic {
  static Eigen::Index unconstrained_rows(Eigen::Index rows) { return rows; }
  static Eigen::Index unconstrained_cols(Eigen::Index cols) { return cols - 1; }
  template <bool Jacobian, typename Mat, typename Lp>
  static auto constrain(const Mat& u, Lp& lp) {
    return stan::math::stochastic_row_constrain<Jacobian>(u, lp);
  }
  static Eigen::MatrixXd unconstrain(const Eigen::MatrixXd& c) {
    return stan::math::stochastic_row_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_stochastic_row(function, name.c_str(), v);
  }
};

/**
 * @brief `sum_to_zero_matrix[n, m]`, from `(n - 1) x (m - 1)` values.
 */
struct sum_to_zero_matrix {
  static Eigen::Index unconstrained_rows(Eigen::Index rows) { return rows - 1; }
  static Eigen::Index unconstrained_cols(Eigen::Index cols) { return cols - 1; }
  template <bool Jacobian, typename Mat, typename Lp>
  static auto constrain(const Mat& u, Lp& lp) {
    return stan::math::sum_to_zero_constrain<Jacobian>(u, lp);
  }
  static Eigen::MatrixXd unconstrain(const Eigen::MatrixXd& c) {
    return stan::math::sum_to_zero_free(c);
  }
  template <typename V>
  static void check(const char* function, const std::string& name,
                    const V& v) {
    stan::math::check_sum_to_zero(function, name.c_str(), v);
  }
};

}  // namespace transform

/**
 * @brief Spec for a constrained vector.
 *
 * @tparam Transform The transform policy.
 */
template <typename Transform>
struct constrained_vector {
  Eigen::Index n;

  /**
   * @brief Construct the spec.
   *
   * @param[in] n The number of elements.
   */
  explicit constrained_vector(Eigen::Index n) : n(n) {}

  template <typename T>
  using value_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  Eigen::Index unconstrained_size() const {
    return Transform::unconstrained_size(n);
  }
  Eigen::Index constrained_size() const { return n; }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp& lp) const {
    return Transform::template constrain<Jacobian>(u, lp);
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return Transform::unconstrain(Eigen::VectorXd(c));
  }

  void suffixes(std::vector<std::string>& out) const {
    for (Eigen::Index i = 1; i <= n; ++i) {
      out.emplace_back("." + std::to_string(i));
    }
  }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    internal::flat_suffixes(unconstrained_size(), out);
  }

  template <typename T>
  value_t<T> zero() const {
    return value_t<T>::Zero(n);
  }

  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    Transform::check(function, name, v);
  }
};

/**
 * @brief Spec for `simplex[n]`.
 */
using simplex = constrained_vector<transform::simplex>;

/**
 * @brief Spec for `ordered[n]`.
 */
using ordered = constrained_vector<transform::ordered>;

/**
 * @brief Spec for `positive_ordered[n]`.
 */
using positive_ordered = constrained_vector<transform::positive_ordered>;

/**
 * @brief Spec for `unit_vector[n]`.
 */
using unit_vector = constrained_vector<transform::unit_vector>;

/**
 * @brief Spec for `sum_to_zero_vector[n]`.
 */
using sum_to_zero = constrained_vector<transform::sum_to_zero>;

/**
 * @brief Spec for a constrained square matrix.
 *
 * @tparam Transform The transform policy.
 */
template <typename Transform>
struct constrained_square_matrix {
  Eigen::Index n;

  /**
   * @brief Construct the spec.
   *
   * @param[in] n The number of rows and columns.
   */
  explicit constrained_square_matrix(Eigen::Index n) : n(n) {}

  template <typename T>
  using value_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  Eigen::Index unconstrained_size() const {
    return Transform::unconstrained_size(n);
  }
  Eigen::Index constrained_size() const { return n * n; }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp& lp) const {
    return Transform::template constrain<Jacobian>(u, n, lp);
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return Transform::unconstrain(internal::to_matrix_colmajor(c, n, n));
  }

  void suffixes(std::vector<std::string>& out) const {
    internal::matrix_suffixes(n, n, out);
  }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    internal::flat_suffixes(unconstrained_size(), out);
  }

  template <typename T>
  value_t<T> zero() const {
    return value_t<T>::Zero(n, n);
  }

  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    Transform::check(function, name, v);
  }
};

/**
 * @brief Spec for `cov_matrix[n]`.
 */
using cov_matrix = constrained_square_matrix<transform::cov_matrix>;

/**
 * @brief Spec for `corr_matrix[n]`.
 */
using corr_matrix = constrained_square_matrix<transform::corr_matrix>;

/**
 * @brief Spec for `cholesky_factor_corr[n]`.
 */
using cholesky_factor_corr =
    constrained_square_matrix<transform::cholesky_factor_corr>;

/**
 * @brief Spec for `cholesky_factor_cov[rows, cols]`, with `rows >= cols`.
 */
struct cholesky_factor_cov {
  Eigen::Index rows;
  Eigen::Index cols;

  /**
   * @brief Construct the spec for a square factor.
   *
   * @param[in] rows The number of rows and columns.
   */
  explicit cholesky_factor_cov(Eigen::Index rows) : rows(rows), cols(rows) {}

  /**
   * @brief Construct the spec.
   *
   * @param[in] rows The number of rows.
   * @param[in] cols The number of columns.
   */
  cholesky_factor_cov(Eigen::Index rows, Eigen::Index cols)
      : rows(rows), cols(cols) {}

  template <typename T>
  using value_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  Eigen::Index unconstrained_size() const {
    return cols + internal::tri(cols) + (rows - cols) * cols;
  }
  Eigen::Index constrained_size() const { return rows * cols; }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp& lp) const {
    return stan::math::cholesky_factor_constrain<Jacobian>(
        u, static_cast<int>(rows), static_cast<int>(cols), lp);
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return stan::math::cholesky_factor_free(
        internal::to_matrix_colmajor(c, rows, cols));
  }

  void suffixes(std::vector<std::string>& out) const {
    internal::matrix_suffixes(rows, cols, out);
  }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    internal::flat_suffixes(unconstrained_size(), out);
  }

  template <typename T>
  value_t<T> zero() const {
    return value_t<T>::Zero(rows, cols);
  }

  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    stan::math::check_cholesky_factor(function, name.c_str(), v);
  }
};

/**
 * @brief Spec for a constrained matrix read from an unconstrained matrix.
 *
 * @tparam Transform The transform policy.
 */
template <typename Transform>
struct constrained_matrix {
  Eigen::Index rows;
  Eigen::Index cols;

  /**
   * @brief Construct the spec.
   *
   * @param[in] rows The number of rows.
   * @param[in] cols The number of columns.
   */
  constrained_matrix(Eigen::Index rows, Eigen::Index cols)
      : rows(rows), cols(cols) {}

  template <typename T>
  using value_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  /**
   * @brief Return the number of rows of the unconstrained matrix.
   */
  Eigen::Index unconstrained_rows() const {
    return Transform::unconstrained_rows(rows);
  }

  /**
   * @brief Return the number of columns of the unconstrained matrix.
   */
  Eigen::Index unconstrained_cols() const {
    return Transform::unconstrained_cols(cols);
  }
  Eigen::Index unconstrained_size() const {
    return unconstrained_rows() * unconstrained_cols();
  }
  Eigen::Index constrained_size() const { return rows * cols; }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp& lp) const {
    return Transform::template constrain<Jacobian>(
        internal::to_matrix_colmajor(u, unconstrained_rows(),
                                     unconstrained_cols()),
        lp);
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    return internal::flatten_colmajor(
        Transform::unconstrain(internal::to_matrix_colmajor(c, rows, cols)));
  }

  void suffixes(std::vector<std::string>& out) const {
    internal::matrix_suffixes(rows, cols, out);
  }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    internal::flat_suffixes(unconstrained_size(), out);
  }

  template <typename T>
  value_t<T> zero() const {
    return value_t<T>::Zero(rows, cols);
  }

  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    Transform::check(function, name, v);
  }
};

/**
 * @brief Spec for `column_stochastic_matrix[rows, cols]`.
 */
using column_stochastic_matrix =
    constrained_matrix<transform::column_stochastic>;

/**
 * @brief Spec for `row_stochastic_matrix[rows, cols]`.
 */
using row_stochastic_matrix = constrained_matrix<transform::row_stochastic>;

/**
 * @brief Spec for `sum_to_zero_matrix[rows, cols]`.
 */
using sum_to_zero_matrix = constrained_matrix<transform::sum_to_zero_matrix>;

/**
 * @brief Spec for `array[n] Inner`.
 *
 * Theta holds each element in turn, first index slowest. The flattened
 * value runs first index fastest, as in a Stan CSV row.
 *
 * @tparam Inner The spec of an element.
 */
template <typename Inner>
struct array_t {
  Eigen::Index n;
  Inner inner;

  template <typename T>
  using value_t = std::vector<typename Inner::template value_t<T>>;

  Eigen::Index unconstrained_size() const {
    return n * inner.unconstrained_size();
  }
  Eigen::Index constrained_size() const { return n * inner.constrained_size(); }

  template <bool Jacobian, typename Vec, typename Lp>
  value_t<typename Vec::Scalar> constrain(const Vec& u, Lp& lp) const {
    value_t<typename Vec::Scalar> out;
    out.reserve(n);
    const Eigen::Index m = inner.unconstrained_size();
    for (Eigen::Index i = 0; i < n; ++i) {
      out.push_back(
          inner.template constrain<Jacobian>(u.segment(i * m, m), lp));
    }
    return out;
  }

  Eigen::VectorXd unconstrain(
      const Eigen::Ref<const Eigen::VectorXd>& c) const {
    const Eigen::Index s = inner.constrained_size();
    const Eigen::Index m = inner.unconstrained_size();
    Eigen::VectorXd out(unconstrained_size());
    Eigen::VectorXd slice(s);
    for (Eigen::Index i = 0; i < n; ++i) {
      for (Eigen::Index k = 0; k < s; ++k) {
        slice(k) = c(i + n * k);
      }
      out.segment(i * m, m) = inner.unconstrain(slice);
    }
    return out;
  }

  void suffixes(std::vector<std::string>& out) const {
    std::vector<std::string> inner_suffixes;
    inner.suffixes(inner_suffixes);
    for (const std::string& s : inner_suffixes) {
      for (Eigen::Index i = 1; i <= n; ++i) {
        out.emplace_back("." + std::to_string(i) + s);
      }
    }
  }

  void unconstrained_suffixes(std::vector<std::string>& out) const {
    std::vector<std::string> inner_suffixes;
    inner.unconstrained_suffixes(inner_suffixes);
    for (Eigen::Index i = 1; i <= n; ++i) {
      for (const std::string& s : inner_suffixes) {
        out.emplace_back("." + std::to_string(i) + s);
      }
    }
  }

  template <typename T>
  value_t<T> zero() const {
    return value_t<T>(n, inner.template zero<T>());
  }

  template <typename V>
  void check(const char* function, const std::string& name, const V& v) const {
    for (std::size_t i = 0; i < v.size(); ++i) {
      inner.check(function, name + "[" + std::to_string(i + 1) + "]", v[i]);
    }
  }
};

/**
 * @brief Return a spec for `array[n] inner`.
 *
 * @tparam Inner The spec of an element.
 * @param[in] n The number of elements.
 * @param[in] inner The spec of an element.
 * @return The spec.
 */
template <typename Inner>
array_t<Inner> array(Eigen::Index n, Inner inner) {
  return array_t<Inner>{n, inner};
}

}  // namespace spec

}  // namespace stancpp
