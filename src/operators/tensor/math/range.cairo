use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

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
>(
    mut start: T, end: T, step: T
) -> Tensor<T> {
    let mut result: Array<T> = array![];
    let zero: T = NumberTrait::zero();
    while !(step >= zero && start >= end)
        && !(step <= zero && start <= end) {
            let v = start;
            result.append(v);
            start += step;
        };

    let shape = array![result.len()];

    TensorTrait::<T>::new(shape.span(), result.span())
}
