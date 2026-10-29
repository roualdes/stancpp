#pragma once

// Stan Math extends Eigen's matrices, which works only when Stan Math is
// included before Eigen.
#if defined(EIGEN_CORE_H) && !defined(EIGEN_MATRIXBASE_PLUGIN)
#error "stancpp: include Stan Math, or stancpp, before any Eigen header"
#endif

#include <stan/math/mix.hpp>

#include <Eigen/Dense>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace stancpp {

/**
 * @brief A view of a column major matrix with contiguous columns.
 *
 * Columns may be any distance apart, so this views a whole matrix, a map of
 * R's memory, or a block of rows. A view never copies, so the viewed memory
 * must outlive it. It is an `Eigen::Map` rather than an `Eigen::Ref`, which
 * silently copies a source it cannot view.
 */
using matrix_view =
    Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>;

/**
 * @brief A view of a contiguous vector with scalar type `Scalar`.
 */
template <typename Scalar>
using vector_view_t =
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;

/**
 * @brief A view of a contiguous vector of doubles.
 */
using vector_view = vector_view_t<double>;

/**
 * @brief A view of a contiguous vector of ints.
 */
using int_vector_view = vector_view_t<int>;

/**
 * @brief Return a view of a matrix.
 *
 * @tparam Derived The type of the matrix.
 * @param[in] x The matrix, which must outlive the view.
 * @param[in] name The name of the matrix, for error messages.
 * @return A view of `x`'s memory.
 * @throw std::invalid_argument If `x`'s columns are not contiguous.
 */
template <typename Derived>
matrix_view view_matrix(const Eigen::DenseBase<Derived>& x, const char* name) {
  static_assert(std::is_same<typename Derived::Scalar, double>::value,
                "view_matrix: data must be double");
  const Eigen::Index row_step =
      Derived::IsRowMajor ? x.outerStride() : x.innerStride();
  const Eigen::Index col_step =
      Derived::IsRowMajor ? x.innerStride() : x.outerStride();
  if (x.rows() > 1 && row_step != 1) {
    throw std::invalid_argument(
        std::string(name) +
        " must be column major with contiguous columns to be viewed without "
        "a copy; transpose or copy it once, before building the model");
  }
  const Eigen::Index stride = x.cols() > 1 ? col_step : x.rows();
  return matrix_view(x.derived().data(), x.rows(), x.cols(),
                     Eigen::OuterStride<>(stride));
}

/**
 * @brief Return a view of a vector.
 *
 * @tparam Derived The type of the vector.
 * @param[in] x The vector, which must outlive the view.
 * @param[in] name The name of the vector, for error messages.
 * @return A view of `x`'s memory.
 * @throw std::invalid_argument If `x` is not a contiguous vector.
 */
template <typename Derived>
vector_view_t<typename Derived::Scalar> view_vector(
    const Eigen::DenseBase<Derived>& x, const char* name) {
  if (x.rows() != 1 && x.cols() != 1) {
    throw std::invalid_argument(std::string(name) + " must be a vector");
  }
  if (x.size() > 1 && x.innerStride() != 1) {
    throw std::invalid_argument(
        std::string(name) +
        " must be contiguous to be viewed without a copy; copy it once, "
        "before building the model");
  }
  return vector_view_t<typename Derived::Scalar>(x.derived().data(),
                                                 x.size());
}

/**
 * @brief Refuse a temporary matrix, which would die before its view.
 *
 * A `const&&` parameter binds every temporary, const or not, and beats the
 * `DenseBase` overloads, which need a conversion. Blocks and maps of
 * longer-lived memory are neither `Eigen::Matrix` nor `Eigen::Array`, so
 * they can still be viewed as temporaries.
 */
template <typename S, int R, int C, int O, int MR, int MC>
matrix_view view_matrix(const Eigen::Matrix<S, R, C, O, MR, MC>&&,
                        const char*) = delete;

/**
 * @brief Refuse a temporary array, which would die before its view.
 */
template <typename S, int R, int C, int O, int MR, int MC>
matrix_view view_matrix(const Eigen::Array<S, R, C, O, MR, MC>&&,
                        const char*) = delete;

/**
 * @brief Refuse a temporary vector, which would die before its view.
 */
template <typename S, int R, int C, int O, int MR, int MC>
vector_view_t<S> view_vector(const Eigen::Matrix<S, R, C, O, MR, MC>&&,
                             const char*) = delete;

/**
 * @brief Refuse a temporary array, which would die before its view.
 */
template <typename S, int R, int C, int O, int MR, int MC>
vector_view_t<S> view_vector(const Eigen::Array<S, R, C, O, MR, MC>&&,
                             const char*) = delete;

}  // namespace stancpp
