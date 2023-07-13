use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::flatten::helpers::_flatten;

/// Cf: TensorTrait::flatten docstring
fn flatten(self: @Tensor<u32>, axis: usize) -> Tensor<u32> {
    _flatten(self, axis)
}
