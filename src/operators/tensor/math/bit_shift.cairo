use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

fn bit_shift<
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
    tensor1: @Tensor<T>, tensor2: @Tensor<T>, direction: felt252
) -> Tensor<T> {
    assert(direction == 'LEFT' || direction == 'RIGHT', 'Unexpected direction.');
    let len = (*tensor2.data).len();
    let mut arr1: Array<T> = array![];
    let mut i: usize = 0;
    let zero: T = NumberTrait::zero();
    let one: T = NumberTrait::one();
    let two: T = one + one;
    while i != len {
        let v = *(*tensor2.data).at(i);
        let mut j: T = zero;
        let mut r: T = one;
        while j != v {
            r = r * two;
            j = j + one;
        };
        arr1.append(r);
        i += 1;
    };
    i = 0;
    let mut arr2: Array<T> = array![];
    while i != len {
        let mut r: T = zero;
        let v1 = *(*tensor1.data).at(i);
        let v2 = *arr1.at(i);
        if (direction == 'LEFT') {
            r = v1 * v2;
        } else {
            r = v1 / v2;
        }
        arr2.append(r);
        i += 1;
    };
    TensorTrait::<T>::new(*tensor1.shape, arr2.span())
}