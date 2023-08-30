use array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: NNTrait::logsoftmax docstring
fn logsoftmax<
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl FTensor: TensorTrait<F, F>,
    impl FDivTensor: Div<Tensor<F>>,
    impl FDrop: Drop<F>
>(
    z: @Tensor<T>, axis: usize
) -> Tensor<F> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;
    let logsoftmax = softmax.log();

    return logsoftmax;
}
