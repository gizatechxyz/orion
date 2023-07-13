use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::flatten::helpers::_flatten;

/// Cf: TensorTrait::flatten docstring
fn flatten(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
    _flatten(self, axis)
}
