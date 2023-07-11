use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::core::Tensor;

/// Cf: TensorTrait::ceil docstring
fn ceil(z: @Tensor<i8>) -> Tensor<i8> {
    return *z;
}
