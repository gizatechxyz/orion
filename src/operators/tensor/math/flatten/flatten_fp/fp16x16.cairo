use orion::numbers::fixed_point::core::FixedType;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::flatten::helpers::_flatten;

/// Cf: TensorTrait::flatten docstring
fn flatten(self: @Tensor<FixedType>, axis: usize) -> Tensor<FixedType> {
    _flatten(self, axis)
}
