use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;
use core::traits::{Into, TryInto};

use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: TensorTrait::exp docstring
fn exp<
    T,
    MAG,
    impl TNumberTrait: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    mut self: Tensor<T>
) -> Tensor<T> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => { result.append((*item).exp()); },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(self.shape, result.span());
}

/// Cf: TensorTrait::exp docstring
fn exp_upcast<
    T,
    TMAG,
    W,
    WMAG,
    impl TFixedTrait: FixedTrait<T, TMAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl WFixedTrait: FixedTrait<W, WMAG>,
    impl WTensor: TensorTrait<W>,
    impl WCopy: Copy<W>,
    impl WDrop: Drop<W>,
    impl TIntoW: Into<T, W>,
>(
    mut self: Tensor<T>
) -> Tensor<W> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => { result.append((TIntoW::into(*item)).exp()); },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(self.shape, result.span());
}
