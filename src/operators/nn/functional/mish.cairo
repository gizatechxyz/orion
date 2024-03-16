use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};

fn mish<
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
    tensor: Tensor<T>
) -> Tensor<T> {
    let exp = tensor.exp();
    let len = (tensor.data).len();
    let mut arr1: Array<T> = array![];
    let mut i: usize = 0;
    while i != len {
        let v = *(exp.data).at(i);
        let r = v + NumberTrait::one();
        arr1.append(r);
        i += 1;
    };
    let log1p = TensorTrait::<T>::new(tensor.shape, arr1.span()).log();
    let tanh = log1p.tanh();

    let mut arr2: Array<T> = array![];
    i = 0;
    while i != len {
        let v1 = *(tensor.data).at(i);
        let v2 = *(tanh.data).at(i);
        let r = v1 * v2;
        arr2.append(r);
        i += 1;
    };
    TensorTrait::<T>::new(tensor.shape, arr2.span())
}
