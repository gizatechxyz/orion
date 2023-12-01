use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{core::{Tensor, TensorTrait, unravel_index}, BoolTensor};
use orion::operators::tensor::helpers::{
    broadcast_shape, broadcast_index_mapping, len_from_shape, check_compatibility
};

/// Cf: TensorTrait::and docstring
fn and(y: @Tensor<bool>, z: @Tensor<bool>) -> Tensor<bool> {
    let broadcasted_shape = broadcast_shape(*y.shape, *z.shape);
    let mut result: Array<bool> = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*y.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*z.shape, indices_broadcasted);

        result.append(*(*y.data)[indices_self] && *(*z.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::new(broadcasted_shape, result.span());
}
