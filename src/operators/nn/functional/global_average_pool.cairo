use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};


fn global_average_pool<
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
    impl TPartialEq: PartialEq<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    tensor: Tensor<T>
) -> Tensor<T> {
    assert((tensor.shape).len() >= 2, 'Unexpected shape.');
    let N = *(tensor.shape).at(0);
    let C = *(tensor.shape).at(1);
    let mut shape = array![N, C];
    let mut i: usize = 2;
    let one: T = NumberTrait::one();
    let len = (tensor.shape).len();
    let mut num: usize = 1;
    let mut num_t: T = one;
    while i < len {
        shape.append(1);
        let v = *(tensor.shape).at(i);
        let mut v_t: T = one;
        let mut j: usize = 1;
        while j != v {
            v_t = v_t + one;
            j += 1;
        };
        num *= v;
        num_t = num_t * v_t;
        i += 1;
    };
    let mut arr: Array<T> = array![];
    i = 0;
    let tensor_len = (tensor.data).len();
    while i < tensor_len {
        let mut j: usize = 0;
        let mut r: T = NumberTrait::zero();
        while j != num {
            r += *(tensor.data).at(i);
            j += 1;
            i += 1;
        };
        arr.append(r / num_t);
    };

    TensorTrait::<T>::new(shape.span(), arr.span())
}
