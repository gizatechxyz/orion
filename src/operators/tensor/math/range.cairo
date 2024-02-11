use core::traits::Into;
use core::traits::TryInto;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;

use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};


fn range<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(mut start: T, end: T, step: T) -> Tensor<T> {
    let mut result: Array<T> = ArrayTrait::<T>::new();
    let zero: T = NumberTrait::zero();
    loop {
        if (step >= zero && start >= end) || (step <= zero && start <= end) {
            break ();
        };
        let v = start;
        result.append(v);
        start += step;
    };
    let shape = array![result.len()];
    return TensorTrait::<T>::new(shape.span(), result.span());
}
