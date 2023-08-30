use core::traits::Into;
use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: NNTrait::leaky_relu docstring
fn leaky_relu_from_int<
    T,
    F,
    INTMAG,
    FPMAG,
    impl TNumber: NumberTrait<T, INTMAG>,
    impl FNumber: NumberTrait<F, FPMAG>,
    impl FFixedTrait: FixedTrait<F, FPMAG>,
    impl FTensor: TensorTrait<F, F>,
    impl MAGInto: Into<INTMAG, FPMAG>,
    impl FMul: Mul<F>,
    impl TPartialOrd: PartialOrd<T>,
    impl FPartialOrd: PartialOrd<F>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>,
    impl FPMAGDrop: Drop<FPMAG>
>(
    mut z: Tensor<T>, alpha: @F
) -> Tensor<F> {
    assert(*alpha < NumberTrait::one(), 'alpha must be less than 1');

    let mut data_result = ArrayTrait::<F>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let fp_current_index = FixedTrait::<F,
                FPMAG>::new_unscaled(((*item).mag()).into(), (*item).is_neg());
                if (*item >= NumberTrait::zero()) {
                    data_result.append(fp_current_index);
                } else {
                    data_result.append(fp_current_index * *alpha);
                };
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span(), z.extra);
}

/// Cf: NNTrait::leaky_relu docstring
fn leaky_relu_from_fp<
    F,
    MAG,
    impl FNumber: NumberTrait<F, MAG>,
    impl FTensor: TensorTrait<F, F>,
    impl FPartialOrd: PartialOrd<F>,
    impl FMul: Mul<F>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>,
>(
    mut z: Tensor<F>, alpha: @F
) -> Tensor<F> {
    assert(*alpha < NumberTrait::one(), 'alpha must be less than 1');

    let mut data_result = ArrayTrait::<F>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                if (*item >= NumberTrait::zero()) {
                    data_result.append(*item);
                } else {
                    data_result.append(*item * *alpha);
                };
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span(), z.extra);
}
