use orion::operators::tensor::core::Tensor;

/// Cf: TensorTrait::ceil docstring
fn ceil(z: @Tensor<u32>) -> Tensor<u32> {
    return *z;
}