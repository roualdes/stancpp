// The data views: they share the viewed memory, refuse layouts they cannot
// view without a copy, and refuse temporaries at compile time.

#include <stancpp/stancpp.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

// Whether an expression of type T can be viewed, rather than refused at
// compile time by a deleted overload.
template <typename T, typename = void>
struct can_view_matrix : std::false_type {};
template <typename T>
struct can_view_matrix<
    T, std::void_t<decltype(stancpp::view_matrix(std::declval<T>(), ""))>>
    : std::true_type {};

template <typename T, typename = void>
struct can_view_vector : std::false_type {};
template <typename T>
struct can_view_vector<
    T, std::void_t<decltype(stancpp::view_vector(std::declval<T>(), ""))>>
    : std::true_type {};

// Named data and blocks of it outlive their views; temporaries do not.
static_assert(can_view_matrix<Eigen::MatrixXd&>::value);
static_assert(can_view_matrix<const Eigen::MatrixXd&>::value);
static_assert(can_view_matrix<const Eigen::ArrayXXd&>::value);
static_assert(
    can_view_matrix<decltype(std::declval<Eigen::MatrixXd&>().middleRows(
        0, 1))>::value);
static_assert(!can_view_matrix<Eigen::MatrixXd>::value);
static_assert(!can_view_matrix<const Eigen::MatrixXd>::value);
static_assert(!can_view_matrix<Eigen::ArrayXXd>::value);
static_assert(!can_view_matrix<const Eigen::ArrayXXd>::value);

static_assert(can_view_vector<Eigen::VectorXi&>::value);
static_assert(can_view_vector<const Eigen::ArrayXi&>::value);
static_assert(!can_view_vector<Eigen::VectorXi>::value);
static_assert(!can_view_vector<const Eigen::VectorXi>::value);
static_assert(!can_view_vector<Eigen::ArrayXi>::value);
static_assert(!can_view_vector<const Eigen::ArrayXi>::value);

TEST(ViewMatrix, SharesMemoryOfABlockOfRows) {
  const Eigen::MatrixXd X = Eigen::MatrixXd::Random(6, 3);
  const stancpp::matrix_view rows = stancpp::view_matrix(X.middleRows(3, 2),
                                                         "X");
  EXPECT_EQ(rows.data(), X.data() + 3);
  EXPECT_EQ(rows, X.middleRows(3, 2));
}

TEST(ViewMatrix, RefusesStridedRows) {
  const Eigen::MatrixXd X = Eigen::MatrixXd::Random(6, 3);
  EXPECT_THROW(stancpp::view_matrix(X.transpose(), "X'"),
               std::invalid_argument);
}

TEST(ViewVector, SharesMemoryOfASegment) {
  const Eigen::VectorXi y = Eigen::VectorXi::LinSpaced(6, 0, 5);
  const stancpp::int_vector_view tail = stancpp::view_vector(y.tail(2), "y");
  EXPECT_EQ(tail.data(), y.data() + 4);
  EXPECT_EQ(tail, y.tail(2));
}

TEST(ViewVector, RefusesARowOfAColumnMajorMatrix) {
  const Eigen::MatrixXd X = Eigen::MatrixXd::Random(6, 3);
  EXPECT_THROW(stancpp::view_vector(X.row(1), "x"), std::invalid_argument);
}

TEST(ViewVector, RefusesAMatrix) {
  const Eigen::MatrixXd X = Eigen::MatrixXd::Random(6, 3);
  EXPECT_THROW(stancpp::view_vector(X, "X"), std::invalid_argument);
}

}  // namespace
