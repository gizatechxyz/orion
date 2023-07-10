use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::flatten::helpers::_flatten;

/// Cf: TensorTrait::flatten docstring
fn flatten(
    self: @Tensor<i8>,
    axis: usize
 ) -> Tensor<i8> {
    _flatten(self, axis)
}
