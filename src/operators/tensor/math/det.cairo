use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

fn det<
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
    self: Tensor<T>
 ) -> Tensor<T> {
    assert((self.shape).len() == 3 && *(self.shape).at(1) == *(self.shape).at(2), 'Unexpected shape.');

    let n = *(self.shape).at(0);

    let mut arr: Array<T> = array![];
    let mut i: usize = 0;

    let mut tensor_array: Array<Tensor<T>> = array![];
    while i < n {
        let s = self.slice(array![i].span(), array![i + 1].span(), Option::Some(array![0].span()), Option::Some(array![1].span()));
        tensor_array.append(TensorTrait::<T>::new(array![*(s.shape).at(1), *(s.shape).at(2)].span(), s.data));
        i += 1;
    };
    i = 0;
    let len = tensor_array.len();
    while i < len {
        let s = tensor_array.at(i);
        let m = determinant(*s);
        arr.append(m);
        i += 1;
    };
    TensorTrait::<T>::new(array![n].span(), arr.span())
}

fn determinant<
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
 ) -> T {
    let shape_len = *(tensor.shape).at(0);
    if shape_len == 2 {
        return *(tensor.data).at(0) * *(tensor.data).at(3) - *(tensor.data).at(1) * *(tensor.data).at(2);
    };
    let zero: T = NumberTrait::zero();
    let one: T = NumberTrait::one();
    let n_one: T = zero - one;
    let mut d = zero;
    let mut j: usize = 0;
    let n = *(tensor.shape).at(0);
    let mut n_j = n_one;

    while j < n {
        n_j = n_j * n_one;
        let cofactor: T = n_j * (*(tensor.data).at(j)) * determinant(minor(tensor, 0, j));
        d = d + cofactor;
        j += 1;
    };
    return d;
 }

fn minor<
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
    tensor: Tensor<T>, i: usize, j: usize
 ) -> Tensor<T> {
    let upper_part = TensorTrait::concat(array![tensor.slice(array![0].span(), array![i].span(), Option::Some(array![0].span()), Option::Some(array![1].span())), tensor.slice(array![i + 1].span(), array![*(tensor.shape).at(0)].span(), Option::Some(array![0].span()), Option::Some(array![1].span()))].span(), 0);
    TensorTrait::concat(array![upper_part.slice(array![0].span(), array![j].span(), Option::Some(array![1].span()), Option::Some(array![1].span())), upper_part.slice(array![j + 1].span(), array![*(tensor.shape).at(1)].span(), Option::Some(array![1].span()), Option::Some(array![1].span()))].span(), 1)
}
