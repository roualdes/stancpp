#pragma once

// Stan Math extends Eigen's matrices, which works only when Stan Math is
// included before Eigen.
#if defined(EIGEN_CORE_H) && !defined(EIGEN_MATRIXBASE_PLUGIN)
#error "stancpp: include Stan Math, or stancpp, before any Eigen header"
#endif

#include <stan/math/mix.hpp>

#include <Eigen/Dense>

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace stancpp {

// Implementation details of the scopes, and a scope that only counts.
namespace internal {

/**
 * @brief Append a value's values in Stan CSV order.
 *
 * @param[in] v The value.
 * @param[out] out The flattened values.
 */
inline void flatten_value(double v, std::vector<double>& out) {
  out.push_back(v);
}

template <typename Derived>
void flatten_value(const Eigen::MatrixBase<Derived>& m,
                   std::vector<double>& out) {
  for (Eigen::Index j = 0; j < m.cols(); ++j) {
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
      out.push_back(stan::math::value_of_rec(m.coeff(i, j)));
    }
  }
}

template <typename Element>
void flatten_value(const std::vector<Element>& v, std::vector<double>& out) {
  if (v.empty()) {
    return;
  }
  std::vector<std::vector<double>> inner(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    flatten_value(v[i], inner[i]);
  }
  for (std::size_t k = 0; k < inner.front().size(); ++k) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      out.push_back(inner[i][k]);
    }
  }
}

/**
 * @brief Fill a value from its flattened values, the inverse of
 * `flatten_value`.
 *
 * @param[in] c The flattened values.
 * @param[out] v The value, already correctly shaped.
 */
inline void unflatten_value(const Eigen::VectorXd& c, double& v) { v = c(0); }

template <int R, int C>
void unflatten_value(const Eigen::VectorXd& c, Eigen::Matrix<double, R, C>& m) {
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < m.cols(); ++j) {
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
      m(i, j) = c(k++);
    }
  }
}

template <typename Element>
void unflatten_value(const Eigen::VectorXd& c, std::vector<Element>& v) {
  const Eigen::Index n = static_cast<Eigen::Index>(v.size());
  if (n == 0) {
    return;
  }
  const Eigen::Index s = c.size() / n;
  Eigen::VectorXd slice(s);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index k = 0; k < s; ++k) {
      slice(k) = c(i + n * k);
    }
    unflatten_value(slice, v[static_cast<std::size_t>(i)]);
  }
}

/**
 * @brief Scope that only counts values.
 *
 * It records no names, so the model methods that return sizes stay cheap;
 * `schema` records the names too.
 */
class sizes {
 public:
  /**
   * @brief The scalar type of the values.
   */
  using scalar_type = double;

  /**
   * @brief Count a parameter and return a zero value of its shape.
   *
   * @tparam Spec The type of the spec.
   * @param[in] spec The parameter's spec.
   * @return A value of zeros.
   */
  template <typename Spec>
  auto declare(const std::string&, const Spec& spec) {
    unconstrained_size_ += spec.unconstrained_size();
    constrained_size_ += spec.constrained_size();
    return spec.template zero<double>();
  }

  /**
   * @brief Count a generated quantity without computing it.
   *
   * @tparam Spec The type of the spec.
   * @tparam F The type of the function computing the value.
   * @param[in] spec The generated quantity's spec.
   * @return A value of zeros.
   */
  template <typename Spec, typename F>
  auto emit(const std::string&, const Spec& spec, F&&) {
    generated_quantities_size_ += spec.constrained_size();
    return spec.template zero<double>();
  }

  /**
   * @brief Return the number of unconstrained values.
   */
  Eigen::Index unconstrained_size() const { return unconstrained_size_; }

  /**
   * @brief Return the number of output values.
   *
   * @param[in] include_gq Whether to count generated quantities.
   * @return The number of values.
   */
  Eigen::Index output_size(bool include_gq) const {
    return constrained_size_ + (include_gq ? generated_quantities_size_ : 0);
  }

 private:
  Eigen::Index unconstrained_size_ = 0;
  Eigen::Index constrained_size_ = 0;
  Eigen::Index generated_quantities_size_ = 0;
};

}  // namespace internal

/**
 * @brief Scope that reads parameters from an unconstrained vector.
 *
 * Each declaration constrains its segment of theta and adds its log
 * absolute Jacobian determinant to the target. Theta is viewed when it is
 * a contiguous vector of scalar type `T` and copied once otherwise.
 *
 * @tparam T The scalar type.
 * @tparam Jacobian Whether to apply the Jacobian adjustment.
 */
template <typename T, bool Jacobian = true>
class reader {
 public:
  /**
   * @brief The scalar type of the values.
   */
  using scalar_type = T;

  /**
   * @brief Construct a reader of theta.
   *
   * @tparam Derived The type of theta.
   * @param[in] theta The unconstrained parameters, which must outlive the
   * reader.
   */
  template <typename Derived>
  explicit reader(const Eigen::MatrixBase<Derived>& theta)
      : size_(theta.size()), lp_(0) {
    if constexpr (viewable<Derived>()) {
      if (contiguous_vector(theta)) {
        data_ = theta.derived().data();
        return;
      }
    }
    owned_.resize(size_);
    for (Eigen::Index i = 0; i < size_; ++i) {
      owned_(i) = theta(i);
    }
    data_ = owned_.data();
    copied_ = true;
  }

  reader(const reader&) = delete;
  reader& operator=(const reader&) = delete;

  /**
   * @brief Read, constrain, and return the next parameter.
   *
   * @tparam Spec The type of the spec.
   * @param[in] name The parameter's name.
   * @param[in] spec The parameter's spec.
   * @return The constrained value.
   * @throw std::invalid_argument If theta is too short.
   */
  template <typename Spec>
  auto declare(const std::string& name, const Spec& spec) {
    const Eigen::Index m = spec.unconstrained_size();
    if (pos_ + m > size_) {
      throw std::invalid_argument(
          "stancpp::reader: ran off the end of theta declaring '" + name +
          "'; theta has " + std::to_string(size_) +
          " elements but the parameter block needs at least " +
          std::to_string(pos_ + m));
    }
    auto value = spec.template constrain<Jacobian>(segment_t(data_ + pos_, m),
                                                   lp_);
    pos_ += m;
    return value;
  }

  /**
   * @brief Return the log density accumulated by the declarations.
   *
   * This is Stan's `target()` at the top of the model block: the log
   * absolute Jacobian determinant, or zero without the adjustment. Call it
   * after the last declaration.
   *
   * @return The accumulated log density.
   * @throw std::invalid_argument If theta was not read exactly.
   */
  const T& target() const {
    if (pos_ != size_) {
      throw std::invalid_argument(
          "stancpp::reader: the parameter block consumed " +
          std::to_string(pos_) + " values but theta has " +
          std::to_string(size_) +
          "; if these are constrained draws, unconstrain them first");
    }
    return lp_;
  }

  /**
   * @brief Return the number of values read.
   */
  Eigen::Index position() const { return pos_; }

  /**
   * @brief Return the size of theta.
   */
  Eigen::Index size() const { return size_; }

  /**
   * @brief Return whether theta was copied rather than viewed.
   */
  bool copied() const { return copied_; }

 private:
  /** A view of one declaration's segment of theta. */
  using segment_t = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

  /** Whether a theta of type `Derived` may be viewed. */
  template <typename Derived>
  static constexpr bool viewable() {
    return std::is_same<typename Derived::Scalar, T>::value &&
           (Derived::Flags & Eigen::DirectAccessBit) != 0;
  }

  /** Whether `x` is a vector with adjacent elements. */
  template <typename Derived>
  static bool contiguous_vector(const Eigen::MatrixBase<Derived>& x) {
    if (x.rows() != 1 && x.cols() != 1) {
      return false;
    }
    if (x.size() <= 1) {
      return true;
    }
    const Eigen::Index row_step =
        Derived::IsRowMajor ? x.outerStride() : x.innerStride();
    const Eigen::Index col_step =
        Derived::IsRowMajor ? x.innerStride() : x.outerStride();
    return (x.cols() == 1 ? row_step : col_step) == 1;
  }

  /** Theta's values, viewed or owned. */
  const T* data_ = nullptr;

  /** The size of theta. */
  Eigen::Index size_;

  /** The copy of theta, when one is needed. */
  Eigen::Matrix<T, Eigen::Dynamic, 1> owned_;

  /** Whether theta was copied. */
  bool copied_ = false;

  /** The number of values read. */
  Eigen::Index pos_ = 0;

  /** The accumulated log density. */
  T lp_;
};

/**
 * @brief Scope that records sizes and Stan CSV names.
 *
 * It reads no values, so it needs no theta.
 */
class schema {
 public:
  /**
   * @brief The scalar type of the values.
   */
  using scalar_type = double;

  /**
   * @brief Record a parameter and return a zero value of its shape.
   *
   * @tparam Spec The type of the spec.
   * @param[in] name The parameter's name.
   * @param[in] spec The parameter's spec.
   * @return A value of zeros.
   */
  template <typename Spec>
  auto declare(const std::string& name, const Spec& spec) {
    std::vector<std::string> suffixes;
    spec.suffixes(suffixes);
    for (const std::string& s : suffixes) {
      constrained_names_.push_back(name + s);
    }
    std::vector<std::string> unconstrained_suffixes;
    spec.unconstrained_suffixes(unconstrained_suffixes);
    for (const std::string& s : unconstrained_suffixes) {
      unconstrained_names_.push_back(name + s);
    }
    entries_.push_back({name, unconstrained_size_, spec.unconstrained_size(),
                        constrained_size_, spec.constrained_size()});
    unconstrained_size_ += spec.unconstrained_size();
    constrained_size_ += spec.constrained_size();
    return spec.template zero<double>();
  }

  /**
   * @brief Record a generated quantity without computing it.
   *
   * @tparam Spec The type of the spec.
   * @tparam F The type of the function computing the value.
   * @param[in] name The generated quantity's name.
   * @param[in] spec The generated quantity's spec.
   * @return A value of zeros.
   */
  template <typename Spec, typename F>
  auto emit(const std::string& name, const Spec& spec, F&&) {
    std::vector<std::string> suffixes;
    spec.suffixes(suffixes);
    for (const std::string& suffix : suffixes) {
      generated_quantities_names_.push_back(name + suffix);
    }
    return spec.template zero<double>();
  }

  /**
   * @brief Where one parameter sits in the unconstrained and constrained
   * layouts.
   */
  struct entry {
    std::string name;
    Eigen::Index unconstrained_offset;
    Eigen::Index unconstrained_size;
    Eigen::Index constrained_offset;
    Eigen::Index constrained_size;
  };

  /**
   * @brief Return the number of unconstrained values.
   */
  Eigen::Index unconstrained_size() const { return unconstrained_size_; }

  /**
   * @brief Return the number of constrained values.
   */
  Eigen::Index constrained_size() const { return constrained_size_; }

  /**
   * @brief Return the parameters' Stan CSV names.
   */
  const std::vector<std::string>& constrained_names() const {
    return constrained_names_;
  }

  /**
   * @brief Return names for the unconstrained values, in theta's order.
   */
  const std::vector<std::string>& unconstrained_names() const {
    return unconstrained_names_;
  }

  /**
   * @brief Return one entry per parameter, in declaration order.
   */
  const std::vector<entry>& entries() const { return entries_; }

  /**
   * @brief Return the number of generated quantity values.
   */
  Eigen::Index generated_quantities_size() const {
    return static_cast<Eigen::Index>(generated_quantities_names_.size());
  }

  /**
   * @brief Return the generated quantities' Stan CSV names.
   */
  const std::vector<std::string>& generated_quantities_names() const {
    return generated_quantities_names_;
  }

  /**
   * @brief Return the number of output values.
   *
   * @param[in] include_gq Whether to count generated quantities.
   * @return The number of values.
   */
  Eigen::Index output_size(bool include_gq) const {
    return constrained_size_ + (include_gq ? generated_quantities_size() : 0);
  }

  /**
   * @brief Return the output names, in Stan's CSV order: parameters, then
   * generated quantities.
   *
   * @param[in] include_gq Whether to include generated quantities.
   * @return The names.
   */
  std::vector<std::string> output_names(bool include_gq = true) const {
    std::vector<std::string> out = constrained_names_;
    if (include_gq) {
      out.insert(out.end(), generated_quantities_names_.begin(),
                 generated_quantities_names_.end());
    }
    return out;
  }

 private:
  Eigen::Index unconstrained_size_ = 0;
  Eigen::Index constrained_size_ = 0;
  std::vector<std::string> constrained_names_;
  std::vector<std::string> unconstrained_names_;
  std::vector<std::string> generated_quantities_names_;
  std::vector<entry> entries_;
};

/**
 * @brief Scope that maps constrained values to an unconstrained vector.
 *
 * Each declaration returns its constrained value, so a later bound that
 * depends on it is unconstrained against the right value.
 */
class unconstrainer {
 public:
  /**
   * @brief The scalar type of the values.
   */
  using scalar_type = double;

  /**
   * @brief Construct an unconstrainer.
   *
   * @param[in] values Each parameter's flattened constrained values, by
   * name, which must outlive the unconstrainer.
   */
  explicit unconstrainer(const std::map<std::string, Eigen::VectorXd>& values)
      : values_(values) {}

  /**
   * @brief Unconstrain the next parameter and return its constrained value.
   *
   * @tparam Spec The type of the spec.
   * @param[in] name The parameter's name.
   * @param[in] spec The parameter's spec.
   * @return The constrained value.
   * @throw std::invalid_argument If the value is missing, the wrong size,
   * or violates its constraint.
   */
  template <typename Spec>
  auto declare(const std::string& name, const Spec& spec) {
    auto it = values_.find(name);
    if (it == values_.end()) {
      throw std::invalid_argument(
          "stancpp::unconstrainer: no value supplied for '" + name + "'");
    }
    if (it->second.size() != spec.constrained_size()) {
      throw std::invalid_argument(
          "stancpp::unconstrainer: '" + name + "' has " +
          std::to_string(it->second.size()) + " values but should have " +
          std::to_string(spec.constrained_size()));
    }
    const Eigen::VectorXd u = spec.unconstrain(it->second);
    if (u.size() != spec.unconstrained_size()) {
      throw std::logic_error("stancpp::unconstrainer: '" + name +
                             "' freed to the wrong size");
    }
    unconstrained_.conservativeResize(unconstrained_.size() + u.size());
    unconstrained_.tail(u.size()) = u;

    auto value = spec.template zero<double>();
    internal::unflatten_value(it->second, value);
    return value;
  }

  /**
   * @brief Return the unconstrained vector.
   */
  const Eigen::VectorXd& unconstrained() const { return unconstrained_; }

 private:
  /** Each parameter's flattened constrained values. */
  const std::map<std::string, Eigen::VectorXd>& values_;

  /** The unconstrained vector. */
  Eigen::VectorXd unconstrained_{0};
};

/**
 * @brief Scope that maps an unconstrained vector to flattened
 * constrained values, Stan's `write_array`.
 */
class writer {
 public:
  /**
   * @brief The scalar type of the values.
   */
  using scalar_type = double;

  /**
   * @brief Construct a writer of theta.
   *
   * @tparam Derived The type of theta.
   * @param[in] theta The unconstrained parameters.
   */
  template <typename Derived>
  explicit writer(const Eigen::MatrixBase<Derived>& theta)
      : theta_(theta.size()) {
    for (Eigen::Index i = 0; i < theta.size(); ++i) {
      theta_(i) = stan::math::value_of_rec(theta(i));
    }
  }

  /**
   * @brief Read, constrain, record, and return the next parameter.
   *
   * @tparam Spec The type of the spec.
   * @param[in] name The parameter's name.
   * @param[in] spec The parameter's spec.
   * @return The constrained value.
   * @throw std::invalid_argument If theta is too short.
   */
  template <typename Spec>
  auto declare(const std::string& name, const Spec& spec) {
    const Eigen::Index m = spec.unconstrained_size();
    if (pos_ + m > theta_.size()) {
      throw std::invalid_argument(
          "stancpp::writer: ran off the end of theta declaring '" + name + "'");
    }
    double lp = 0;
    auto value = spec.template constrain<false>(
        Eigen::VectorXd(theta_.segment(pos_, m)), lp);
    pos_ += m;
    internal::flatten_value(value, constrained_);
    return value;
  }

  /**
   * @brief Compute, check, record, and return a generated quantity.
   *
   * @tparam Spec The type of the spec.
   * @tparam F The type of the function computing the value.
   * @param[in] name The generated quantity's name.
   * @param[in] spec The generated quantity's spec.
   * @param[in] f The function computing the value.
   * @return The value.
   * @throw std::invalid_argument If the value has the wrong size.
   * @throw std::domain_error If the value violates its constraint.
   */
  template <typename Spec, typename F>
  auto emit(const std::string& name, const Spec& spec, F&& f) {
    auto value = f();
    const std::size_t before = generated_quantities_.size();
    internal::flatten_value(value, generated_quantities_);
    const Eigen::Index produced =
        static_cast<Eigen::Index>(generated_quantities_.size() - before);
    if (produced != spec.constrained_size()) {
      throw std::invalid_argument(
          "stancpp::writer: '" + name + "' produced " +
          std::to_string(produced) + " values but its spec says " +
          std::to_string(spec.constrained_size()));
    }
    spec.check("stancpp::writer", name, value);
    return value;
  }

  /**
   * @brief Return the flattened generated quantities.
   */
  Eigen::VectorXd generated_quantities() const {
    return Eigen::Map<const Eigen::VectorXd>(
        generated_quantities_.data(),
        static_cast<Eigen::Index>(generated_quantities_.size()));
  }

  /**
   * @brief Return the flattened constrained parameters.
   *
   * @throw std::invalid_argument If theta was not read exactly.
   */
  Eigen::VectorXd constrained() const {
    if (pos_ != theta_.size()) {
      throw std::invalid_argument(
          "stancpp::writer: the parameter block consumed " +
          std::to_string(pos_) + " values but theta has " +
          std::to_string(theta_.size()));
    }
    return Eigen::Map<const Eigen::VectorXd>(
        constrained_.data(), static_cast<Eigen::Index>(constrained_.size()));
  }

 private:
  /** Theta, as doubles. */
  Eigen::VectorXd theta_;

  /** The number of values read. */
  Eigen::Index pos_ = 0;

  /** The flattened constrained parameters. */
  std::vector<double> constrained_;

  /** The flattened generated quantities. */
  std::vector<double> generated_quantities_;
};

}  // namespace stancpp
