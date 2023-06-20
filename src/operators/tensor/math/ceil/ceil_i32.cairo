use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::core::Tensor;

/// Cf: TensorTrait::ceil docstring
fn ceil(z: @Tensor<i32>) -> Tensor<i32> {
    return *z;
}
