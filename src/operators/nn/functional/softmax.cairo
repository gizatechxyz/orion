use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::math::{exp::exp_upcast, arithmetic::div_downcast};

/// Cf: NNTrait::softmax docstring
fn softmax<
    T,
    impl TTensor: TensorTrait<T>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    z: @Tensor<T>, axis: Option<i32>
) -> Tensor<T> {
    let axis = match axis {
        Option::Some(val) => val,
        Option::None => -1
    };

    let exp_tensor = z.exp();
    let sum = exp_tensor
        .reduce_sum(Option::Some(array![axis].span()), Option::Some(true), Option::Some(false));

    exp_tensor / sum
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
    z: @Tensor<T>, axis: Option<i32>
) -> Tensor<T> {
    let axis = match axis {
        Option::Some(val) => val,
        Option::None => -1
    };

    let exp_tensor: Tensor<W> = exp_upcast(*z);
    let sum = exp_tensor
        .reduce_sum(Option::Some(array![axis].span()), Option::Some(true), Option::Some(false));

    div_downcast(@exp_tensor, @sum)
}

