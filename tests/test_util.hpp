#pragma once

// What the tests share: reading JSON into Eigen, reading a model's
// reference file, and comparing vectors and matrices.

#include <stancpp/stancpp.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace test {

/**
 * @brief Read a JSON file.
 *
 * @param[in] path The file.
 * @return Its contents.
 * @throw std::runtime_error If the file cannot be opened.
 */
inline nlohmann::json read_json(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  return nlohmann::json::parse(in);
}

/**
 * @brief Read a model's reference file, tests/reference/<name>.json.
 *
 * It holds the model's data, its test point `theta`, and CmdStan's values
 * there; tests/reference/update.sh writes them.
 *
 * @param[in] name The model's name.
 * @return The reference.
 */
inline nlohmann::json read_reference(const std::string& name) {
  return read_json(std::string(TESTS_DIR) + "/reference/" + name + ".json");
}

/**
 * @brief Read a JSON array of numbers into a vector.
 *
 * @tparam Scalar The type of the elements.
 * @param[in] j The array.
 * @return The vector.
 */
template <typename Scalar = double>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_vector(const nlohmann::json& j) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out(j.size());
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = j[i].get<Scalar>();
  }
  return out;
}

/**
 * @brief Read a JSON array of rows into a matrix.
 *
 * @param[in] j The rows, each an array of the same length.
 * @return The matrix.
 */
inline Eigen::MatrixXd to_matrix(const nlohmann::json& j) {
  const Eigen::Index rows = j.size();
  const Eigen::Index cols = rows > 0 ? j[0].size() : 0;
  Eigen::MatrixXd out(rows, cols);
  for (Eigen::Index i = 0; i < rows; ++i) {
    for (Eigen::Index k = 0; k < cols; ++k) {
      out(i, k) = j[i][k].get<double>();
    }
  }
  return out;
}

/**
 * @brief Expect two vectors or matrices to agree to a relative tolerance.
 *
 * The tolerance is relative to the larger of 1 and the largest magnitude in
 * `expected`.
 *
 * @param[in] actual The values computed.
 * @param[in] expected The values expected.
 * @param[in] tolerance The relative tolerance.
 */
template <typename A, typename B>
void expect_near(const Eigen::MatrixBase<A>& actual,
                 const Eigen::MatrixBase<B>& expected, double tolerance) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  if (actual.size() == 0) {
    return;
  }
  const double scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
  EXPECT_LE((actual - expected).cwiseAbs().maxCoeff(), tolerance * scale)
      << "actual   " << actual.transpose() << "\nexpected "
      << expected.transpose();
}

/**
 * @brief Expect two numbers to agree to a relative tolerance.
 *
 * @param[in] actual The value computed.
 * @param[in] expected The value expected.
 * @param[in] tolerance The tolerance, relative to the larger of 1 and
 * `|expected|`.
 */
inline void expect_near(double actual, double expected, double tolerance) {
  EXPECT_NEAR(actual, expected,
              tolerance * std::max(1.0, std::abs(expected)));
}

}  // namespace test
