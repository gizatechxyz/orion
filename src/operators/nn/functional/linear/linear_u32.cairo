use array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32, u32TensorAdd};
use orion::performance::core::PerfomanceTrait;
use orion::performance::implementations::impl_performance_u32::Performance_u32;

/// Cf: NNTrait::linear docstring
fn linear_u32(z: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>) -> Tensor<u32> {
    assert(z.shape.len() == 1, 'input tensor must be 1D');
    assert(weights.shape.len() == 2, 'weights tensor must be 2D');
    assert(bias.shape.len() == 1, 'bias tensor must be 1D');

    let dot = weights.matmul(@z);
    let sum = dot + bias;

    return sum;
}
