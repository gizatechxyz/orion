use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};

fn lrn<
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
    tensor: Tensor<T>, size: usize, alpha: Option<T>, beta: Option<T>, bias: Option<T>
) -> Tensor<T> {
    assert((tensor.shape).len() >= 2, 'Unexpected shape.');

    let minc = *(tensor.shape).at(1);

    let zero: T = NumberTrait::zero();
    let one: T = NumberTrait::one();
    let two: T = one + one;
    let three: T = two + one;
    let ten: T = three.pow(two) + one;
    let four: T = two + two;
    let n_10000: T = ten.pow(zero - four);
    let alpha_v = match alpha {
        Option::Some(val) => val,
        Option::None => one / n_10000,
    };
    let beta_v = match beta {
        Option::Some(val) => val,
        Option::None => three / (two * two),
    };
    let bias_v = match bias {
        Option::Some(val) => val,
        Option::None => one,
    };
    let c1: usize = (size - 1) / 2;
    let mut c2: usize = ((size - 1) / 2) + 1;
    if (c1 * 2 + 1 != size) {
        c2 += 1;
    };
    let mut c: usize = 0;
    let mut sum_array: Array<Tensor<T>> = array![];
    while c != minc {
        let mut begin = 0;
        if c > c1 {
            begin = c - c1;
        };
        let end = NumberTrait::min(minc, c + c2);
        let s = tensor.slice(array![begin].span(), array![end].span(), Option::Some(array![1].span()), Option::Some(array![1].span()));
        sum_array.append(s.reduce_sum(axis: 1, keepdims: true));
        c += 1;
    };
    let sum_concat = TensorTrait::concat(sum_array.span(), 1);
    let len = (tensor.data).len();
    let mut arr: Array<T> = array![];
    let mut i: usize = 0;
    let mut size_t: T = zero;
    while i != size {
        size_t = size_t + one;
        i += 1;
    };
    i = 0;
    while i != len {
        let v1 = *(sum_concat.data).at(i);
        let v2 = *(tensor.data).at(i);
        let r = v2 / ((bias_v + (alpha_v / size_t) * v1 * v1).pow(beta_v));
        arr.append(r);
        i += 1;
    };
    TensorTrait::<T>::new(tensor.shape, arr.span())
}
