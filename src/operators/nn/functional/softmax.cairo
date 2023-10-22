use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::math::{exp::exp_upcast, arithmetic::div_downcast};
use orion::numbers::fixed_point::core::FixedTrait;

/// Cf: NNTrait::softmax docstring
fn softmax<
    T,
    impl TTensor: TensorTrait<T>,
    impl TTensor: TensorTrait<T>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    z: @Tensor<T>, axis: usize
) -> Tensor<T> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;

    return softmax;
}

/// Cf: NNTrait::softmax docstring
fn softmaxWide<
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
    let softmax: Tensor<T> = div_downcast(@exp_tensor, @sum);

    return softmax;
}

use orion::numbers::{FP16x16, FP16x16W};
use orion::operators::tensor::{
    implementations::tensor_fp16x16wide::{FP16x16WTensor, FP16x16WTensorDiv}, FP16x16Tensor
};
use debug::PrintTrait;

/// Cf: NNTrait::softmax docstring
fn softmaxWide2(z: @Tensor<FP16x16>, axis: usize) -> Tensor<FP16x16> {
    let exp_tensor: Tensor<FP16x16W> = exp_upcast(*z);
    (exp_tensor.data.len()).print();

    // let sum = exp_tensor.reduce_sum(axis, true);
    // (*sum.data.at(0)).print();
    // let softmax = exp_tensor / sum;
    // return exp_tensor;
    *z
}
