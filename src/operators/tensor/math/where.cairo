use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, unravel_index};
use orion::operators::tensor::helpers::{
    broadcast_shape, broadcast_index_mapping, len_from_shape, check_compatibility
};

/// Cf: TensorTrait::where docstring
fn where<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TFTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, x: @Tensor<T>, y: @Tensor<T>
) -> Tensor<T> {
    let xy_shape = broadcast_shape(*x.shape, *y.shape);
    let broadcasted_shape = broadcast_shape(*self.shape, xy_shape);
    let mut result: Array<T> = array![];

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    while n != num_elements {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_cond = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_x = broadcast_index_mapping(*x.shape, indices_broadcasted);
        let indices_y = broadcast_index_mapping(*y.shape, indices_broadcasted);

        let res = NumberTrait::where(
            *(*self.data)[indices_cond], *(*x.data)[indices_x], *(*y.data)[indices_y]
        );
        result.append(res);

        n += 1;
    };

    TensorTrait::new(broadcasted_shape, result.span())
}
