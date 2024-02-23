use core::traits::Into;
use core::traits::TryInto;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;

use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};


fn blackman_window<
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
    let two: T = one_step + one_step;
    let three: T = two + one_step;
    let n25: T = three.pow(three) - two;
    let alpha: T = (n25 - two * two) / (n25 * two);
    let beta: T = two / n25;
    let n_0_5: T = (one_step - two) / two;

    let ni = TensorTrait::range(start, size, one_step);
    assert!((ni.shape).len() == 1, "Unexpected shape 1.");
    let mut N_1 = size;
    if periodic != Option::Some(1) {
        N_1 = N_1 - one_step;
    };
    let len = *(ni.shape).at(0);
    let mut arr1: Array<T> = ArrayTrait::<T>::new();
    let mut i: usize = 0;
    loop {
        let v = *(ni.data).at(i);
        let r = (v * (PI * two)) / N_1;
        arr1.append(r);
        i += 1;
        if i >= len {
            break ();
        };
    };
    let window_cos = TensorTrait::<T>::new(ni.shape, arr1.span()).cos();
    i = 0;
    let mut a1: Array<T> = ArrayTrait::<T>::new();
    loop {
        let v = *(window_cos.data).at(i);
        let r = v * n_0_5;
        a1.append(r);
        i += 1;
        if i >= len {
            break ();
        };
    };
    let window1 = TensorTrait::<T>::new(ni.shape, a1.span());

    let mut arr2: Array<T> = ArrayTrait::<T>::new();
    i = 0;
    loop {
        let v = *(ni.data).at(i);
        let r = v * (PI * two * two) / N_1;
        arr2.append(r);
        i += 1;
        if i >= len {
            break ();
        };
    };
    let window_cos_2 = TensorTrait::<T>::new(ni.shape, arr2.span()).cos();

    let mut a2: Array<T> = ArrayTrait::<T>::new();
    i = 0;
    loop {
        let v = *(window_cos_2.data).at(i);
        let r = v * beta + alpha;
        a2.append(r);
        i += 1;
        if i >= len {
            break ();
        };
    };
    let window2 = TensorTrait::<T>::new(ni.shape, a2.span());

    let mut arr: Array<T> = ArrayTrait::<T>::new();
    i = 0;
    loop {
        let v1 = *(window1.data).at(i);
        let v2 = *(window2.data).at(i);
        let r = v1 + v2;
        arr.append(r);
        i += 1;
        if i >= len {
            break ();
        };
    };
    return TensorTrait::<T>::new(ni.shape, arr.span());
}
