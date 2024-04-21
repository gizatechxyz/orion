use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::numbers::fixed_point::core::FixedTrait;

fn square<
    T,
    MAG,
    impl FTensorTrait: TensorTrait<T>,
    impl FNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    self: @Tensor<T>
) -> Tensor<T> {
    let mut data = *self.data;
    let mut output_data = array![];

    loop {
        match data.pop_front() {
            Option::Some(item) => {
                let ele = *item;
                output_data.append(ele * ele);
            },
            Option::None => { break; }
        };
    };

    let tensor_square = TensorTrait::new(*self.shape, output_data.span());

    tensor_square
}

/// Cf: TensorTrait::reduce_sum_square docstring
fn reduce_sum_square<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, axis: usize, keepdims: bool
) -> Tensor<T> {
    let tensor_square = square(self);
    let tensor_square_sum = tensor_square
        .reduce_sum(
            Option::Some(array![axis.try_into().unwrap()].span()),
            Option::Some(keepdims),
            Option::Some(false)
        );

    tensor_square_sum
}
