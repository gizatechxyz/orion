use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::math::reduce_sum_single_axis::reduce_sum_single_axis;

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
    let tensor_square_sum = reduce_sum_single_axis(@tensor_square, axis: axis, keepdims: keepdims);

    tensor_square_sum
}
