#pragma once

// Stan Math extends Eigen's matrices, which works only when Stan Math is
// included before Eigen.
#if defined(EIGEN_CORE_H) && !defined(EIGEN_MATRIXBASE_PLUGIN)
#error "stancpp: include Stan Math, or stancpp, before any Eigen header"
#endif

#include <stan/math/mix.hpp>

#include <Eigen/Dense>

#include <boost/random/additive_combine.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "stancpp/scope.hpp"

namespace stancpp {

// Implementation details of the model methods.
namespace internal {

/**
 * @brief Return a row or column vector as a column vector of doubles.
 *
 * @tparam Derived The type of the vector.
 * @param[in] x The vector.
 * @return The column vector.
 */
template <typename Derived>
Eigen::VectorXd as_column(const Eigen::MatrixBase<Derived>& x) {
  Eigen::VectorXd out(x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    out(i) = x(i);
  }
  return out;
}

/**
 * @brief Throw if a size is not the one expected.
 *
 * @param[in] what The name of the argument.
 * @param[in] got The size given.
 * @param[in] want The size expected.
 * @throw std::invalid_argument If `got != want`.
 */
inline void check_size(const char* what, Eigen::Index got, Eigen::Index want) {
  if (got != want) {
    throw std::invalid_argument(std::string("stancpp: ") + what + " has size " +
                                std::to_string(got) + " but should have " +
                                std::to_string(want));
  }
}

/**
 * @brief Whether `Model` declares generated quantities with
 * `generated_quantities()`.
 */
template <typename Model, typename Scope, typename Parameters, typename RNG,
          typename = void>
struct has_generated_quantities : std::false_type {};

template <typename Model, typename Scope, typename Parameters, typename RNG>
struct has_generated_quantities<
    Model, Scope, Parameters, RNG,
    std::void_t<decltype(std::declval<const Model&>().generated_quantities(
        std::declval<Scope&>(), std::declval<const Parameters&>(),
        std::declval<RNG&>()))>> : std::true_type {};

/**
 * @brief The generator handed to `generated_quantities()` when only names
 * and sizes are wanted. No value is computed then, so it is never drawn
 * from.
 */
using placeholder_rng = boost::ecuyer1988;

/**
 * @brief Have a model declare its blocks to a scope, in Stan's order.
 *
 * A model writes each block once, as declarations made to a scope: each
 * parameter with `scope.declare(name, spec)`, and each generated quantity
 * with `scope.emit(name, spec, f)`. The scope decides what a declaration
 * does: `schema` records names and sizes, and `writer` records output
 * values. This function is the one place that calls the blocks in order,
 * passing the parameters on:
 *  1. `parameters(scope)`, which returns the parameters `p`, then
 *  2. `generated_quantities(scope, p, rng)`, when the model has it.
 *
 * @tparam Model The type of the model.
 * @tparam Scope The type of the scope.
 * @tparam RNG The type of the random number generator.
 * @param[in] model The model.
 * @param[in,out] scope The scope.
 * @param[in,out] rng The random number generator.
 */
template <typename Model, typename Scope, typename RNG>
void declare_blocks(const Model& model, Scope& scope, RNG& rng) {
  const auto p = model.parameters(scope);
  using Parameters = std::decay_t<decltype(p)>;
  if constexpr (has_generated_quantities<Model, Scope, Parameters,
                                         RNG>::value) {
    model.generated_quantities(scope, p, rng);
  }
}

/**
 * @brief Return a model's schema: its names and sizes, with its
 * generated quantities.
 *
 * @tparam Model The type of the model.
 * @param[in] model The model.
 * @return The schema.
 */
template <typename Model>
schema model_schema(const Model& model) {
  schema s;
  placeholder_rng rng;
  declare_blocks(model, s, rng);
  return s;
}

/**
 * @brief Return a model's sizes, with its generated quantities, without its
 * names.
 *
 * @tparam Model The type of the model.
 * @param[in] model The model.
 * @return The sizes.
 */
template <typename Model>
sizes model_sizes(const Model& model) {
  sizes s;
  placeholder_rng rng;
  declare_blocks(model, s, rng);
  return s;
}

/**
 * @brief Return the constrained parameters for theta.
 *
 * @tparam Model The type of the model.
 * @tparam Derived The type of theta.
 * @param[in] model The model.
 * @param[in] theta The unconstrained parameters.
 * @return The values, in Stan's CSV order.
 */
template <typename Model, typename Derived>
Eigen::VectorXd constrain_values(const Model& model,
                                 const Eigen::MatrixBase<Derived>& theta) {
  writer w(theta);
  model.parameters(w);
  return w.constrained();
}

/**
 * @brief Return the constrained parameters then the generated quantities
 * for theta, one Stan CSV row.
 *
 * @tparam Model The type of the model.
 * @tparam Derived The type of theta.
 * @tparam RNG The type of the random number generator.
 * @param[in] model The model.
 * @param[in] theta The unconstrained parameters.
 * @param[in,out] rng The random number generator.
 * @return The values, in Stan's CSV order.
 */
template <typename Model, typename Derived, typename RNG>
Eigen::VectorXd constrain_output(const Model& model,
                                 const Eigen::MatrixBase<Derived>& theta,
                                 RNG& rng) {
  writer w(theta);
  declare_blocks(model, w, rng);
  const Eigen::VectorXd c = w.constrained();
  const Eigen::VectorXd gq = w.generated_quantities();
  Eigen::VectorXd out(c.size() + gq.size());
  out << c, gq;
  return out;
}

/**
 * @brief Return theta for one set of constrained parameters.
 *
 * @tparam Model The type of the model.
 * @tparam Derived The type of the constrained parameters.
 * @param[in] model The model.
 * @param[in] s The model's schema.
 * @param[in] theta The constrained parameters, in Stan's CSV order.
 * @return The unconstrained parameters.
 * @throw std::invalid_argument If `theta` has the wrong size.
 * @throw std::domain_error If `theta` violates a constraint.
 */
template <typename Model, typename Derived>
Eigen::VectorXd unconstrain(const Model& model, const schema& s,
                            const Eigen::MatrixBase<Derived>& theta) {
  check_size("theta", theta.size(), s.constrained_size());
  std::map<std::string, Eigen::VectorXd> values;
  for (const schema::entry& e : s.entries()) {
    values[e.name] = theta.segment(e.constrained_offset, e.constrained_size);
  }
  unconstrainer u(values);
  model.parameters(u);
  return u.unconstrained();
}

}  // namespace internal

/**
 * @brief Base class giving a model BridgeStan's methods.
 *
 * A model derives from `model<Derived>` and provides
 *  - `parameters(scope)`, its parameter declarations,
 *  - `log_prob<Propto, Jacobian>(theta)`, its log density on the
 *    unconstrained scale, templated on the scalar type of theta, which
 *    passes `Propto` to each of its `foo_lpdf<Propto>` calls, and
 *  - optionally, `generated_quantities(scope, parameters, rng)`, its
 *    generated quantities, templated on the type of `rng`.
 *
 * Each generated quantity is declared with `scope.emit(name, spec, f)`,
 * where `f` computes the value only when it is needed, and the value is
 * checked against its spec's constraint. Generated quantities are all a
 * model reports beyond its parameters, including what Stan would call
 * transformed parameters; a value that `log_prob` also needs comes from a
 * helper that both call.
 *
 * The log density methods take runtime `propto` and `jacobian` flags, both
 * `true` by default as in BridgeStan; with `propto`, terms that do not
 * depend on the parameters are dropped. Each method that returns a vector
 * or matrix has an overload that writes into caller-provided buffers.
 * Errors are reported by exception:
 * `std::invalid_argument` for wrongly sized or shaped arguments, and
 * `std::domain_error` when Stan Math rejects a value.
 *
 * @tparam Derived The model.
 */
template <typename Derived>
class model {
 public:
  /**
   * @brief Return the number of constrained parameters.
   *
   * @param[in] include_gq Whether to count generated quantities.
   * @return The number of values.
   */
  Eigen::Index param_num(bool include_gq = false) const {
    return internal::model_sizes(self()).output_size(include_gq);
  }

  /**
   * @brief Return the number of unconstrained parameters.
   *
   * @return The size of theta.
   */
  Eigen::Index param_unc_num() const {
    return internal::model_sizes(self()).unconstrained_size();
  }

  /**
   * @brief Return the constrained parameters' names, in Stan's CSV order.
   *
   * @param[in] include_gq Whether to include generated quantities.
   * @return The names.
   */
  std::vector<std::string> param_names(bool include_gq = false) const {
    return internal::model_schema(self()).output_names(include_gq);
  }

  /**
   * @brief Return names for the unconstrained parameters, in theta's order.
   *
   * @return The names.
   */
  std::vector<std::string> param_unc_names() const {
    return internal::model_schema(self()).unconstrained_names();
  }

  /**
   * @brief Return the constrained parameters for theta.
   *
   * @tparam D The type of theta.
   * @param[in] theta_unc The unconstrained parameters.
   * @return The values, ordered as `param_names()`.
   */
  template <typename D>
  Eigen::VectorXd param_constrain(
      const Eigen::MatrixBase<D>& theta_unc) const {
    return internal::constrain_values(self(), theta_unc);
  }

  /**
   * @brief Write the constrained parameters for theta.
   *
   * @tparam D The type of theta.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[out] theta The values, ordered as `param_names()`.
   * @throw std::invalid_argument If `theta` has the wrong size.
   */
  template <typename D>
  void param_constrain(const Eigen::MatrixBase<D>& theta_unc,
                       Eigen::Ref<Eigen::VectorXd> theta) const {
    internal::check_size("theta", theta.size(), param_num());
    theta = param_constrain(theta_unc);
  }

  /**
   * @brief Return the constrained parameters for theta, then optionally
   * the generated quantities, drawn using `rng`.
   *
   * @tparam D The type of theta.
   * @tparam RNG The type of the random number generator.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[in] include_gq Whether to include generated quantities.
   * @param[in,out] rng The random number generator, drawn from only when
   * `include_gq`.
   * @return The values, ordered as `param_names(include_gq)`.
   * @throw std::domain_error If a generated quantity violates its
   * constraint.
   */
  template <typename D, typename RNG>
  Eigen::VectorXd param_constrain(const Eigen::MatrixBase<D>& theta_unc,
                                  bool include_gq, RNG& rng) const {
    return include_gq ? internal::constrain_output(self(), theta_unc, rng)
                      : internal::constrain_values(self(), theta_unc);
  }

  /**
   * @brief Write the constrained parameters for theta, then optionally the
   * generated quantities, drawn using `rng`.
   *
   * @tparam D The type of theta.
   * @tparam RNG The type of the random number generator.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[out] theta The values, ordered as `param_names(include_gq)`.
   * @param[in] include_gq Whether to include generated quantities.
   * @param[in,out] rng The random number generator, drawn from only when
   * `include_gq`.
   * @throw std::invalid_argument If `theta` has the wrong size.
   * @throw std::domain_error If a generated quantity violates its
   * constraint.
   */
  template <typename D, typename RNG>
  void param_constrain(const Eigen::MatrixBase<D>& theta_unc,
                       Eigen::Ref<Eigen::VectorXd> theta, bool include_gq,
                       RNG& rng) const {
    internal::check_size("theta", theta.size(), param_num(include_gq));
    theta = param_constrain(theta_unc, include_gq, rng);
  }

  /**
   * @brief Return theta for constrained parameters.
   *
   * @tparam D The type of the constrained parameters.
   * @param[in] theta The constrained parameters, ordered as `param_names()`.
   * @return The unconstrained parameters.
   * @throw std::invalid_argument If `theta` has the wrong size.
   * @throw std::domain_error If `theta` violates a constraint.
   */
  template <typename D>
  Eigen::VectorXd param_unconstrain(const Eigen::MatrixBase<D>& theta) const {
    return internal::unconstrain(self(), internal::model_schema(self()),
                                 theta);
  }

  /**
   * @brief Write theta for constrained parameters.
   *
   * @tparam D The type of the constrained parameters.
   * @param[in] theta The constrained parameters, ordered as `param_names()`.
   * @param[out] theta_unc The unconstrained parameters.
   * @throw std::invalid_argument If an argument has the wrong size.
   * @throw std::domain_error If `theta` violates a constraint.
   */
  template <typename D>
  void param_unconstrain(const Eigen::MatrixBase<D>& theta,
                         Eigen::Ref<Eigen::VectorXd> theta_unc) const {
    internal::check_size("theta_unc", theta_unc.size(), param_unc_num());
    theta_unc = param_unconstrain(theta);
  }

  /**
   * @brief Return the log density.
   *
   * @tparam D The type of theta.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[in] propto Whether to drop terms that do not depend on the
   * parameters.
   * @param[in] jacobian Whether to apply the Jacobian adjustment.
   * @return The log density.
   */
  template <typename D>
  double log_density(const Eigen::MatrixBase<D>& theta_unc, bool propto = true,
                     bool jacobian = true) const {
    if (!propto) {
      return jacobian ? self().template log_prob<false, true>(theta_unc)
                      : self().template log_prob<false, false>(theta_unc);
    }
    // With doubles, every term looks constant and `propto` would drop them
    // all, so evaluate with autodiff variables, as BridgeStan does.
    stan::math::nested_rev_autodiff nested;
    const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> x =
        internal::as_column(theta_unc);
    return jacobian ? self().template log_prob<true, true>(x).val()
                    : self().template log_prob<true, false>(x).val();
  }

  /**
   * @brief Return the log density and its gradient.
   *
   * @tparam D The type of theta.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[in] propto Whether to drop terms that do not depend on the
   * parameters.
   * @param[in] jacobian Whether to apply the Jacobian adjustment.
   * @return The log density and the gradient.
   */
  template <typename D>
  std::pair<double, Eigen::VectorXd> log_density_gradient(
      const Eigen::MatrixBase<D>& theta_unc, bool propto = true,
      bool jacobian = true) const {
    Eigen::VectorXd grad(theta_unc.size());
    const double lp = log_density_gradient(theta_unc, grad, propto, jacobian);
    return {lp, std::move(grad)};
  }

  /**
   * @brief Return the log density and write its gradient.
   *
   * @tparam D The type of theta.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[out] grad The gradient.
   * @param[in] propto Whether to drop terms that do not depend on the
   * parameters.
   * @param[in] jacobian Whether to apply the Jacobian adjustment.
   * @return The log density.
   * @throw std::invalid_argument If `grad` has the wrong size.
   */
  template <typename D>
  double log_density_gradient(const Eigen::MatrixBase<D>& theta_unc,
                              Eigen::Ref<Eigen::VectorXd> grad,
                              bool propto = true, bool jacobian = true) const {
    internal::check_size("grad", grad.size(), theta_unc.size());
    const Eigen::VectorXd x = internal::as_column(theta_unc);
    double lp = 0;
    dispatch(propto, jacobian, [&](auto f) {
      stan::math::gradient(f, x, lp, grad.data(), grad.data() + grad.size());
    });
    return lp;
  }

  /**
   * @brief Return the log density, its gradient, and its Hessian.
   *
   * @tparam D The type of theta.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[in] propto Whether to drop terms that do not depend on the
   * parameters.
   * @param[in] jacobian Whether to apply the Jacobian adjustment.
   * @return The log density, the gradient, and the Hessian.
   */
  template <typename D>
  std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> log_density_hessian(
      const Eigen::MatrixBase<D>& theta_unc, bool propto = true,
      bool jacobian = true) const {
    Eigen::VectorXd grad(theta_unc.size());
    Eigen::MatrixXd hess(theta_unc.size(), theta_unc.size());
    const double lp =
        log_density_hessian(theta_unc, grad, hess, propto, jacobian);
    return {lp, std::move(grad), std::move(hess)};
  }

  /**
   * @brief Return the log density and write its gradient and Hessian.
   *
   * @tparam D The type of theta.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[out] grad The gradient.
   * @param[out] hess The Hessian.
   * @param[in] propto Whether to drop terms that do not depend on the
   * parameters.
   * @param[in] jacobian Whether to apply the Jacobian adjustment.
   * @return The log density.
   * @throw std::invalid_argument If an output has the wrong size.
   */
  template <typename D>
  double log_density_hessian(const Eigen::MatrixBase<D>& theta_unc,
                             Eigen::Ref<Eigen::VectorXd> grad,
                             Eigen::Ref<Eigen::MatrixXd> hess,
                             bool propto = true, bool jacobian = true) const {
    internal::check_size("grad", grad.size(), theta_unc.size());
    internal::check_size("hess rows", hess.rows(), theta_unc.size());
    internal::check_size("hess cols", hess.cols(), theta_unc.size());
    const Eigen::VectorXd x = internal::as_column(theta_unc);
    double lp = 0;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    dispatch(propto, jacobian,
             [&](auto f) { stan::math::hessian(f, x, lp, g, H); });
    grad = g;
    hess = H;
    return lp;
  }

  /**
   * @brief Return the log density and the product of its Hessian with `v`.
   *
   * @tparam D The type of theta.
   * @tparam V The type of the vector.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[in] v The vector.
   * @param[in] propto Whether to drop terms that do not depend on the
   * parameters.
   * @param[in] jacobian Whether to apply the Jacobian adjustment.
   * @return The log density and the product.
   */
  template <typename D, typename V>
  std::pair<double, Eigen::VectorXd> log_density_hessian_vector_product(
      const Eigen::MatrixBase<D>& theta_unc, const Eigen::MatrixBase<V>& v,
      bool propto = true, bool jacobian = true) const {
    Eigen::VectorXd hvp(theta_unc.size());
    const double lp =
        log_density_hessian_vector_product(theta_unc, v, hvp, propto, jacobian);
    return {lp, std::move(hvp)};
  }

  /**
   * @brief Return the log density and write the product of its Hessian
   * with `v`.
   *
   * @tparam D The type of theta.
   * @tparam V The type of the vector.
   * @param[in] theta_unc The unconstrained parameters.
   * @param[in] v The vector.
   * @param[out] hvp The product.
   * @param[in] propto Whether to drop terms that do not depend on the
   * parameters.
   * @param[in] jacobian Whether to apply the Jacobian adjustment.
   * @return The log density.
   * @throw std::invalid_argument If `v` or `hvp` has the wrong size.
   */
  template <typename D, typename V>
  double log_density_hessian_vector_product(
      const Eigen::MatrixBase<D>& theta_unc, const Eigen::MatrixBase<V>& v,
      Eigen::Ref<Eigen::VectorXd> hvp, bool propto = true,
      bool jacobian = true) const {
    internal::check_size("v", v.size(), theta_unc.size());
    internal::check_size("hvp", hvp.size(), theta_unc.size());
    const Eigen::VectorXd x = internal::as_column(theta_unc);
    const Eigen::VectorXd w = internal::as_column(v);
    double lp = 0;
    Eigen::VectorXd Hv;
    dispatch(propto, jacobian, [&](auto f) {
      stan::math::hessian_times_vector(f, x, w, lp, Hv);
    });
    hvp = Hv;
    return lp;
  }

 private:
  /** The model. */
  const Derived& self() const { return static_cast<const Derived&>(*this); }

  /** The log density as a functor of theta, for Stan Math. */
  template <bool Propto, bool Jacobian>
  auto functor() const {
    return [this](const auto& x) {
      return self().template log_prob<Propto, Jacobian>(x);
    };
  }

  /** Call `body` with the functor for the runtime flags. */
  template <typename Body>
  void dispatch(bool propto, bool jacobian, Body&& body) const {
    if (propto) {
      jacobian ? body(functor<true, true>()) : body(functor<true, false>());
    } else {
      jacobian ? body(functor<false, true>()) : body(functor<false, false>());
    }
  }
};

/**
 * @brief Return theta for each row of a matrix of constrained draws.
 *
 * @tparam Model The type of the model.
 * @tparam Derived The type of the draws.
 * @param[in] model The model.
 * @param[in] draws One draw per row, columns as `model.param_names()`.
 * @return One unconstrained draw per row.
 * @throw std::invalid_argument If the draws have the wrong number of
 * columns.
 * @throw std::domain_error If a draw violates a constraint.
 */
template <typename Model, typename Derived>
Eigen::MatrixXd unconstrain_draws(const Model& model,
                                  const Eigen::MatrixBase<Derived>& draws) {
  const schema s = internal::model_schema(model);
  if (draws.cols() != s.constrained_size()) {
    throw std::invalid_argument(
        "stancpp::unconstrain_draws: draws has " +
        std::to_string(draws.cols()) + " columns but the model has " +
        std::to_string(s.constrained_size()) + " constrained parameters");
  }
  Eigen::MatrixXd out(draws.rows(), s.unconstrained_size());
  for (Eigen::Index r = 0; r < draws.rows(); ++r) {
    out.row(r) = internal::unconstrain(model, s, draws.row(r)).transpose();
  }
  return out;
}

}  // namespace stancpp
