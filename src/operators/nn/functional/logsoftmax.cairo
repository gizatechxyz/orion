use array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::math::{exp::exp_upcast, arithmetic::div_downcast};

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

/// Cf: NNTrait::logsoftmax docstring
fn logsoftmaxWide<
    T,
    TMAG,
    W,
    WMAG,
    impl TTensor: TensorTrait<T>,
    impl WTensor: TensorTrait<W>,
    impl TDiv: Div<T>,
    impl TIntoW: Into<T, W>,
    impl WTryIntoT: TryInto<W, T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl WCopy: Copy<W>,
    impl WDrop: Drop<W>,
    impl TFixed: FixedTrait<T, TMAG>,
    impl WFixed: FixedTrait<W, WMAG>,
>(
    z: @Tensor<T>, axis: usize
) -> Tensor<T> {
    let exp_tensor: Tensor<W> = exp_upcast(*z);
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = div_downcast(@exp_tensor, @sum);
    softmax.log()
}