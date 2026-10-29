#pragma once

#include <stan/math/mix.hpp>

#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

/**
 * @brief Helpers that read the examples' data files into Eigen.
 *
 * The data files are CmdStan's JSON format, so each example's `.json` file
 * also runs its `.stan` file.
 */
namespace examples {

/**
 * @brief Read a JSON data file.
 *
 * @param[in] path The path to the file.
 * @return The parsed JSON.
 * @throw std::invalid_argument If the file cannot be opened.
 */
inline nlohmann::json read_data(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument("cannot open " + path);
  }
  return nlohmann::json::parse(in);
}

/**
 * @brief Return a JSON array of numbers as a vector.
 *
 * @tparam Scalar The scalar type of the vector.
 * @param[in] x The array, such as a Stan `vector[N]` or `array[N] int`.
 * @return The vector.
 * @throw std::invalid_argument If `x` is not an array.
 */
template <typename Scalar = double>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_vector(const nlohmann::json& x) {
  if (!x.is_array()) {
    throw std::invalid_argument("to_vector: expected an array");
  }
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    out(i) = x[i].get<Scalar>();
  }
  return out;
}

/**
 * @brief Return a JSON array of rows as a matrix.
 *
 * CmdStan writes both a `matrix[N, K]` and an `array[N] vector[K]` as `N`
 * rows of `K` numbers.
 *
 * @param[in] x The array of rows.
 * @return The `N` by `K` matrix.
 * @throw std::invalid_argument If `x` is not an array of equal length rows.
 */
inline Eigen::MatrixXd to_matrix(const nlohmann::json& x) {
  if (!x.is_array() || x.empty() || !x[0].is_array()) {
    throw std::invalid_argument("to_matrix: expected an array of rows");
  }
  Eigen::MatrixXd out(x.size(), x[0].size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (x[i].size() != x[0].size()) {
      throw std::invalid_argument("to_matrix: rows differ in length");
    }
    out.row(i) = to_vector(x[i]).transpose();
  }
  return out;
}

}  // namespace examples
