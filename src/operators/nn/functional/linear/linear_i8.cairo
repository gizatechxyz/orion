use array::SpanTrait;

use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i8::{Tensor_i8, i8TensorAdd};
use orion::performance::core::PerfomanceTrait;

/// Cf: NNTrait::linear docstring
fn linear_i8(
    z: Tensor<i8>, weights: Tensor<i8>, bias: Tensor<i8>
) -> Tensor<i8> {
    assert(z.shape.len() == 1, 'input tensor must be 1D');
    assert(weights.shape.len() == 2, 'weights tensor must be 2D');
    assert(bias.shape.len() == 1, 'bias tensor must be 1D');

    let dot = weights.matmul(@z);
    let sum = dot + bias;

    return sum;
}
