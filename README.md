# `StanCpp`: Stan models in C++

`StanCpp` provides tools to write a [Stan](https://mc-stan.org/) model
directly in C++ using only [Stan
Math](https://github.com/stan-dev/math). Many features of Stan are
included, but I've probably missed some. A `StanCpp` model has nearly
the same API as a [BridgeStan](https://github.com/roualdes/bridgestan)
model.

## Why

This project was vibed as an after thought of trying to calculate the
[Posterior Averaging Information
Criterion (PAIC)](https://www.mdpi.com/1099-4300/25/3/468) using Stan.

The PAIC requires the gradient and Hessian of the log density, one
observation at a time, and the log likelihood (no prior). `StanCpp`
enables such calculations with one model, without recompiling, and
without making copies of the data.

Writing `StanCpp` models is certainly less convenient than writing a
Stan program. There are some benefits.

A `StanCpp` model is compiled once and uses views of caller owned
data.  Because of this, model creation is cheap.  For data of a
different size, e.g. one row of a model matrix, just create a new
model object.

Data can come from any source so long as it lives in Eigen data
structures by the time it hits a `StanCpp` model. Because everything
is C++, input and output formats are up to you.

Custom functions follow the same design principles for Stan's built-in
functions, just use `stan::math::` operators instead of `std::`
operators, so that the autodiff knows how to deal with them.

## Components of a `StanCpp` Model

Parameters are defined in a `stancpp::model` method named `parameters`.

```cpp
template <typename Scope>
auto parameters(Scope& scope) const {
  return std::tuple{scope.declare("B", spec::matrix(P, K)),
                    scope.declare("Sigma", spec::cov_matrix(K)),
                    scope.declare("tau", spec::lb(spec::real{}, 0.0))};
}
```

Parameters are used to calculate a log density function.

```cpp
template <bool Propto, bool Jacobian, typename Derived>
auto log_prob(const Eigen::MatrixBase<Derived>& theta) const {
  using T = typename Derived::Scalar;
  stancpp::reader<T, Jacobian> scope(theta);
  auto [B, Sigma, tau] = parameters(scope);

  T target = scope.target();
  target += stan::math::lognormal_lpdf<Propto>(tau, 0.0, 1.0);
  ...
  return target;
}
```

A model that derives from `stancpp::model` has many of BridgeStan's
methods. Like BridgeStan, `propto` and `jacobian` default to `true`.

```cpp
const auto [lp, grad] = model.log_density_gradient(theta);
const Eigen::VectorXd draw = model.param_constrain(theta);
const std::vector<std::string> names = model.param_names();
```

Generated quantities are produced only when requested.

```cpp
template <typename Scope, typename Parameters, typename RNG>
auto generated_quantities(Scope& scope, const Parameters& p, RNG& rng) const {
  const auto& [alpha, beta] = p;
  auto y_rep = scope.emit("y_rep", spec::array(N, spec::integer{}), [&] {
    return stan::math::bernoulli_logit_rng(eta(alpha, beta), rng);
  });
  return std::tuple{y_rep};
}
```

`param_constrain`'s second argument dictates whether or not generated
quantities are produced.

```cpp
std::mt19937 rng(1);
const Eigen::VectorXd row = model.param_constrain(theta, true, rng);
const std::vector<std::string> names = model.param_names(true);
```

## Examples

* A Bernoulli example shows the basics.

* The [paic](./examples/paic.cpp) shows calculations one observation
at a time, prior weighting (as required by PAIC), and a poor
implementation of Newton's algorithm and a MH sampler.

* A custom function with a hand-written gradient is in
  [functions.cpp](./examples/functions.cpp).

* The [mvn](./examples/mvn.cpp) model demonstrates generated
  quantities and transformed data.

The [examples/ directory](./examples/) has more. Each example is one
file, such as `logistic.cpp`, holding the model and a short `main()`
that reads its data from `logistic.json`. The same data works with the
corresponding Stan file, e.g. `logistic.stan`. The examples are built
with the rest of the project, which fetches Stan Math and
[nlohmann/json](https://github.com/nlohmann/json).

```sh
cmake -S . -B build
cmake --build build --parallel
./build/examples/logistic
```

## Using `StanCpp` in a C++ project

The `StanCpp` library is header-only, requires C++20, and requires
Stan Math, which it does not fetch.

If your project uses CMake, link the `stancpp::stancpp` target
alongside the target that provides Stan Math.

```cmake
target_link_libraries(my_model INTERFACE my_stan_math stancpp::stancpp)
```

Otherwise, add the `include/` folder of this repository to your
include paths. Then include

```cpp
#include <stancpp/stancpp.hpp>
```

## For developers

The tests use [GoogleTest](https://github.com/google/googletest) and run
with ctest.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Configuring this repository as the top-level project turns on the tests and
examples, `STANCPP_BUILD_TESTS` and `STANCPP_BUILD_EXAMPLES`, which fetch
Stan Math, GoogleTest, and nlohmann/json. A project that adds `StanCpp` as a
subdirectory gets neither and fetches nothing.

The tests check every parameter type against its own definition, and each
model in `tests/models/` against values from CmdStan stored in
`tests/reference/`, so they need no CmdStan to run. After changing a model,
rebuild its reference values with `tests/reference/update.sh`, with
`CMDSTAN` set to a CmdStan 2.40 or newer, which shares the simplex transform
of Stan Math 5.4.

## Licensing

The project is distributed under the [MIT License](./LICENSE).
