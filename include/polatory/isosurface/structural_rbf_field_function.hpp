#pragma once

#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/structural/interpolant.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

class StructuralRbfFieldFunction : public FieldFunction {
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();

 public:
  explicit StructuralRbfFieldFunction(structural::StructuralInterpolant3& interpolant,
                                      double accuracy = kInfinity)
      : interpolant_(interpolant), accuracy_(accuracy) {}

  VecX operator()(const geometry::Points3& points) const override {
    // The regular Polatory lattice evaluates the field one working batch at a
    // time and releases old lattice layers. Preparing every local structural
    // interpolant for the complete model bbox defeats that streaming behavior
    // and can multiply evaluator memory by the number of structural domains.
    // Prepare the local evaluators for the current lattice batch instead.
    return interpolant_.evaluate(points, accuracy_);
  }

  void set_evaluation_bbox(const geometry::Bbox3& /*bbox*/) override {
    // Intentionally deferred to operator(). See the comment above. This keeps
    // fine-resolution structural meshing on the same bounded-memory path as the
    // original single-interpolant Polatory workflow.
  }

 private:
  structural::StructuralInterpolant3& interpolant_;
  double accuracy_;
};

}  // namespace polatory::isosurface
