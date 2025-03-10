use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};


fn mean<
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
    args: Span<Tensor<T>>
) -> Tensor<T> {
    let len = args.len();
    let mut i: usize = 1;
    let mut t = *args.at(0);
    let mut len_t: T = NumberTrait::one();
    while i != len {
        let v = *args.at(i);
        t = t + v;
        len_t += NumberTrait::one();
        i += 1;
    };

    let mut arr: Array<T> = array![];
    let count = (t.data).len();
    i = 0;
    while i != count {
        let v = *(t.data).at(i);
        let r = v / len_t;
        arr.append(r);
        i += 1;
    };
    TensorTrait::<T>::new(t.shape, arr.span())
}
