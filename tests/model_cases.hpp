#pragma once

// One case per test model: its reference file, the data it views, its test
// point, and the model itself. A case owns the data its model views, so it
// is never copied or moved.
//
// Cases whose likelihood splits into units also give `units()`, the number
// of units, and `unit(n, prior_weight)` and `whole(prior_weight)`, models of
// unit n and of all the data.

#include <stancpp/stancpp.hpp>

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "models/bounds.hpp"
#include "models/hier.hpp"
#include "models/logistic.hpp"
#include "models/mix.hpp"
#include "models/mvn.hpp"
#include "test_util.hpp"

namespace test {

/**
 * @brief What every case holds, read from tests/reference/<name>.json.
 */
struct case_base {
  explicit case_base(const char* name)
      : ref(read_reference(name)), theta(to_vector(ref["theta"])) {}
  case_base(const case_base&) = delete;
  case_base& operator=(const case_base&) = delete;

  nlohmann::json ref;
  Eigen::VectorXd theta;
};

struct logistic_case : case_base {
  static constexpr const char* name = "logistic";
  // Output names of values drawn at random, which CmdStan's draw cannot
  // match.
  inline static const std::vector<std::string> random = {};

  logistic_case()
      : case_base(name),
        X(to_matrix(ref["data"]["X"])),
        y(to_vector<int>(ref["data"]["y"])),
        model(X, y, ref["data"]["prior_weight"].get<double>()) {}

  Eigen::Index units() const { return X.rows(); }
  models::LogisticPosterior unit(Eigen::Index n, double prior_weight) const {
    return {X.middleRows(n, 1), y.segment(n, 1), prior_weight};
  }
  models::LogisticPosterior whole(double prior_weight) const {
    return {X, y, prior_weight};
  }

  Eigen::MatrixXd X;
  Eigen::VectorXi y;
  models::LogisticPosterior model;
};

struct mvn_case : case_base {
  static constexpr const char* name = "mvn";
  inline static const std::vector<std::string> random = {"Y_rep"};

  // The model views one observation per column, so Y is transposed once.
  mvn_case()
      : case_base(name),
        X(to_matrix(ref["data"]["X"])),
        Yt(to_matrix(ref["data"]["Y"]).transpose()),
        model(X, Yt, ref["data"]["prior_weight"].get<double>()) {}

  Eigen::Index units() const { return X.rows(); }
  models::MvnPosterior unit(Eigen::Index n, double prior_weight) const {
    return {X.middleRows(n, 1), Yt.middleCols(n, 1), prior_weight};
  }
  models::MvnPosterior whole(double prior_weight) const {
    return {X, Yt, prior_weight};
  }

  Eigen::MatrixXd X;
  Eigen::MatrixXd Yt;
  models::MvnPosterior model;
};

struct mix_case : case_base {
  static constexpr const char* name = "mix";
  inline static const std::vector<std::string> random = {};

  mix_case()
      : case_base(name),
        y(to_vector(ref["data"]["y"])),
        C(ref["data"]["C"].get<Eigen::Index>()),
        model(y, C, ref["data"]["prior_weight"].get<double>()) {}

  Eigen::Index units() const { return y.size(); }
  models::MixPosterior unit(Eigen::Index n, double prior_weight) const {
    return {y.segment(n, 1), C, prior_weight};
  }
  models::MixPosterior whole(double prior_weight) const {
    return {y, C, prior_weight};
  }

  Eigen::VectorXd y;
  Eigen::Index C;
  models::MixPosterior model;
};

struct hier_case : case_base {
  static constexpr const char* name = "hier";
  inline static const std::vector<std::string> random = {};

  // The model views one group per column, so Zobs is transposed once.
  hier_case()
      : case_base(name),
        Zt(to_matrix(ref["data"]["Zobs"]).transpose()),
        model(Zt) {}

  Eigen::MatrixXd Zt;
  models::HierPosterior model;
};

struct bounds_case : case_base {
  static constexpr const char* name = "bounds";
  inline static const std::vector<std::string> random = {};

  bounds_case()
      : case_base(name),
        y(to_vector(ref["data"]["y"])),
        L(to_vector(ref["data"]["L"])),
        U(to_vector(ref["data"]["U"])),
        model(y, L, U) {}

  Eigen::VectorXd y;
  Eigen::VectorXd L;
  Eigen::VectorXd U;
  models::BoundsPosterior model;
};

/**
 * @brief Names each typed test by its case's model.
 */
struct case_names {
  template <typename Case>
  static std::string GetName(int) {
    return Case::name;
  }
};

}  // namespace test
