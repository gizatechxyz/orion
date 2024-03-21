use orion::numbers::NumberTrait;
use orion::operators::tensor::{core::{Tensor, TensorTrait, unravel_index}, BoolTensor, U32Tensor};
use orion::operators::tensor::helpers::{
    broadcast_shape, broadcast_index_mapping, len_from_shape, check_compatibility
};

/// Cf: TensorTrait::and docstring
fn and(y: @Tensor<bool>, z: @Tensor<bool>) -> Tensor<usize> {
    let broadcasted_shape = broadcast_shape(*y.shape, *z.shape);
    let mut result: Array<usize> = array![];

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    while n != num_elements {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*y.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*z.shape, indices_broadcasted);

        let r = if *(*y.data)[indices_self] && *(*z.data)[indices_other] {
            1
        } else {
            0
        };

        result.append(r);

        n += 1;
    };

    TensorTrait::new(broadcasted_shape, result.span())
}
