use orion::operators::tensor::core::Tensor;

/// Cf: TensorTrait::abs docstring
fn abs(z: @Tensor<u32>) -> Tensor<u32> {
    return *z;
}
