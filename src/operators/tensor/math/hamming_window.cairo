use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

fn hamming_window<
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
    let n46: T = n25 * two - two * two;
    let alpha: T = n25 / n46;

    let beta: T = one_step - alpha;
    let ni = TensorTrait::range(start, size, one_step);
    assert((ni.shape).len() == 1, 'Unexpected shape 1.');
    let mut N_1 = size;

    if periodic != Option::Some(1) {
        N_1 = N_1 - one_step;
    };

    let len = *(ni.shape).at(0);
    let mut arr: Array<T> = array![];
    let mut i: usize = 0;
    while i != len {
        let v = *(ni.data).at(i);
        let r = v * PI * two / N_1;
        arr.append(r);
        i += 1;
    };

    let window = TensorTrait::<T>::new(ni.shape, arr.span());
    let window_cos = window.cos();
    let len2 = *(ni.shape).at(0);
    let mut arr2: Array<T> = array![];
    let mut j: usize = 0;
    while j != len2 {
        let v = *(window_cos.data).at(j);
        let v_2 = alpha - v * beta;
        arr2.append(v_2);
        j += 1;
    };

    let window_cos_2 = TensorTrait::<T>::new(ni.shape, arr2.span());

    window_cos_2
}
