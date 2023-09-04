use array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: NNTrait::logsoftmax docstring
fn logsoftmax<
    T, impl TTensor: TensorTrait<T>, impl TDivTensor: Div<Tensor<T>>, impl TDrop: Drop<T>
>(
    z: @Tensor<T>, axis: usize
) -> Tensor<T> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;
    let logsoftmax = softmax.log();

    return logsoftmax;
}
