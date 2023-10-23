use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{Into, TryInto};

use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: TensorTrait::exp docstring
fn exp<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl FTensor: TensorTrait<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    mut self: Tensor<T>
) -> Tensor<T> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                result.append((*item).exp());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(self.shape, result.span());
}

use debug::PrintTrait;

/// Cf: TensorTrait::exp docstring
fn exp_upcast<
    T,
    MAG,
    W,
    WMAG,
    impl TFixedTrait: FixedTrait<T, MAG>,
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
            Option::Some(item) => {
                result.append((TIntoW::into(*item)).exp());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(self.shape, result.span());
}
