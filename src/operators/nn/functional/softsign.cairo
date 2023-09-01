use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: NNTrait::softsign docstring
fn softsign_from_int<
    T,
    F,
    INTMAG,
    FPMAG,
    impl TNumber: NumberTrait<T, INTMAG>,
    impl FFixedTrait: FixedTrait<F, FPMAG>,
    impl FTensor: TensorTrait<F, F>,
    impl MAGInto: Into<INTMAG, FPMAG>,
    impl FAdd: Add<F>,
    impl FDiv: Div<F>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>,
    impl FPMAGDrop: Drop<FPMAG>
>(
    mut z: Tensor<T>
) -> Tensor<F> {
    let mut data_result = ArrayTrait::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let fp_current_item = FixedTrait::<F,
                FPMAG>::new_unscaled(((*item).mag()).into(), (*item).is_neg());
                let result = fp_current_item / (FixedTrait::ONE() + fp_current_item.abs());
                data_result.append(result);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span());
}

/// Cf: NNTrait::softsign docstring
fn softsign_from_fp<
    F,
    MAG,
    impl FTensor: TensorTrait<F, F>,
    impl FFixed: FixedTrait<F, MAG>,
    impl FPartialOrd: PartialOrd<F>,
    impl FAdd: Add<F>,
    impl FDiv: Div<F>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>,
>(
    mut z: Tensor<F>
) -> Tensor<F> {
    let mut data_result = ArrayTrait::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let result = *item / (FixedTrait::ONE() + (*item).abs());
                data_result.append(result);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span());
}
