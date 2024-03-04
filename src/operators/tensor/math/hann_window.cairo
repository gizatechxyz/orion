use core::traits::Into;
use core::traits::TryInto;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;

use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};


fn hann_window<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    size: T, PI: T, periodic: Option<usize>
) -> Tensor<T> {
    let start: T = NumberTrait::zero();
    let one_step: T = NumberTrait::one();
    let ni = TensorTrait::range(start, size, one_step);
    assert!((ni.shape).len() == 1, "Unexpected shape 1.");
    let mut N_1 = size;
    if periodic != Option::Some(1) {
        N_1 = N_1 - one_step;
    };
    let len = *(ni.shape).at(0);
    let mut arr: Array<T> = ArrayTrait::<T>::new();
    let mut i: usize = 0;
    loop {
        let v = *(ni.data).at(i);
        let r = v * PI / N_1;
        arr.append(r);
        i += 1;
        if i >= len {
            break ();
        };
    };
    let window = TensorTrait::<T>::new(ni.shape, arr.span());
    let window_sin = window.sin();
    let len2 = *(ni.shape).at(0);
    let mut arr2: Array<T> = ArrayTrait::<T>::new();
    let mut j: usize = 0;
    loop {
        let v = *(window_sin.data).at(j);
        let v_2 = v * v;
        arr2.append(v_2);
        j += 1;
        if j >= len2 {
            break ();
        };
    };
    let window_sin_2 = TensorTrait::<T>::new(ni.shape, arr2.span());
    return window_sin_2;
}
