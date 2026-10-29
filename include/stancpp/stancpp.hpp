#pragma once

// All of stancpp: data views, specs, scopes, and the model base class.
//
// Include this first, before any Eigen header. It brings in Stan Math, with
// its reverse, forward and mixed mode autodiff, and then Eigen, in the order
// they must come: Stan Math extends Eigen's matrices.

#include "stancpp/data.hpp"
#include "stancpp/spec.hpp"
#include "stancpp/scope.hpp"
#include "stancpp/model.hpp"

// What models use by convention. The headers above include these only for
// their own use, so include them here for the models:
//  - <tuple>, since models return their declarations as a std::tuple, and
//  - <utility>, since models pass their data to view_matrix and view_vector
//    with std::forward, so that a temporary is refused rather than viewed.
#include <tuple>
#include <utility>
