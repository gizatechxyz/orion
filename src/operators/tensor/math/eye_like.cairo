use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

fn eye_like<
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
    self: @Tensor<T>, k: Option<i32>
) -> Tensor<T> {
    assert((*self.shape).len() == 2 || (*self.shape).len() == 1, 'Unexpected shape.');
    let mut shape = *self.shape;
    if (*self.shape).len()==1 {
        shape = (array![*(*self.shape).at(0), *(*self.shape).at(0)]).span();
    };
    let M = *shape.at(1);
    let K = match k {
        Option::Some(val) => val,
        Option::None => 0,
    };
    let len = *(shape.at(0)) * (*(shape.at(1)));
    let mut i: usize = 0;
    let mut arr: Array<T> = array![];
    while i != len {
        arr.append(NumberTrait::zero());
        i += 1;
    };
    if (K >= M.try_into().unwrap()) {
        return TensorTrait::<T>::new(shape, arr.span());
    };
    let mut j: usize = 0;
    if (K < 0){
        j = (-(K)).try_into().unwrap() * M;
    } else {
        j = K.try_into().unwrap();
    };
    let end: usize = (M.try_into().unwrap() - K).try_into().unwrap() * M;
    i = 0;
    arr = array![];
    while i != len {
        if (i == j && j < end) {
            arr.append(NumberTrait::one());
            j += (M + 1);
        } else {
            arr.append(NumberTrait::zero());
        };
        i += 1;
    };
    return TensorTrait::<T>::new(shape, arr.span());
}