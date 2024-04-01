use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

fn tile<
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
>(self: Tensor<T>, repeats: Span<usize>) -> Tensor<T> {
    let mut tensor = self;
    let len = (tensor.shape).len();
    let mut i: usize = 0;
    while i != len {
        let mut k = len - i - 1;
        let mut arr: Array<Tensor<T>> = array![];
        let mut j: usize = 0;
        if (*repeats.at(k) == 0) {
            tensor = TensorTrait::<T>::new(array![0].span(), array![].span());
            i = len;
        } else {
            while j != *repeats.at(k) {
                arr.append(tensor);
                j += 1;
            };
            if (arr.len() > 1) {
                tensor = TensorTrait::concat(arr.span(), k);
            }
            i += 1;
        }
    };
    tensor
}