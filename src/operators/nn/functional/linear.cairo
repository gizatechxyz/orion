use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: NNTrait::linear docstring
fn linear<
    T,
    impl TTensor: TensorTrait<T>,
    impl TAddTensor: Add<Tensor<T>>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    z: Tensor<T>, weights: Tensor<T>, bias: Tensor<T>
) -> Tensor<T> {
    assert(z.shape.len() == 1, 'input tensor must be 1D');
    assert(weights.shape.len() == 2, 'weights tensor must be 2D');
    assert(bias.shape.len() == 1, 'bias tensor must be 1D');

    let dot = weights.matmul(@z);
    let sum = dot + bias;

    sum
}
