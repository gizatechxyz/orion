use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::exp docstring
fn exp_from_int<
    T,
    F,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl FFixedTrait: FixedTrait<F, MAG>,
    impl FTensor: TensorTrait<F, F>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>,
    impl MDrop: Drop<MAG>
>(
    mut self: Tensor<T>
) -> Tensor<F> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                result
                    .append(
                        FixedTrait::<F, MAG>::new_unscaled((*item).mag(), (*item).is_neg()).exp()
                    );
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(self.shape, result.span(), self.extra);
}

fn exp_from_fp<
    F,
    MAG,
    impl FFixedTrait: FixedTrait<F, MAG>,
    impl FTensor: TensorTrait<F, F>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>,
>(
    mut self: Tensor<F>
) -> Tensor<F> {
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

    return TensorTrait::new(self.shape, result.span(), self.extra);
}
