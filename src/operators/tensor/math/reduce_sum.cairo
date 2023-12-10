use core::option::OptionTrait;
use core::array::ArrayTrait;
use core::array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};


/// Cf: TensorTrait::reduce_sum docstring
fn reduce_sum<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, axis: usize, keepdims: bool
) -> Tensor<T> {
    let mut output_data = ArrayTrait::new();

    if (*self.shape).len() == 1 {
        assert(axis == 0, 'axis out of dimensions');
        let current_sum = accumulate_sum::<T>(*self.data, *self.shape, *self.shape, axis);
        output_data.append(current_sum);

        let mut output_shape = ArrayTrait::new();
        output_shape.append(1);

        return TensorTrait::new(output_shape.span(), output_data.span());
    } else {
        assert(axis <= (*self.shape).len(), 'axis out of dimensions');
        let output_shape = reduce_output_shape(*self.shape, axis, false);
        let output_data_len = len_from_shape(output_shape);
        let mut index: usize = 0;
        loop {
            let output_indices = unravel_index(index, output_shape);
            let current_sum = accumulate_sum::<T>(*self.data, *self.shape, output_indices, axis);

            output_data.append(current_sum);

            index += 1;
            if index == output_data_len {
                break ();
            };
        };

        if keepdims {
            let output_shape = reduce_output_shape(*self.shape, axis, true);
            return TensorTrait::<T>::new(output_shape, output_data.span());
        } else {
            return TensorTrait::<T>::new(output_shape, output_data.span());
        }
    }
}


/// Helper function that accumulates the sum of elements along a specific axis.
///
/// # Arguments
/// * `input_data` - The input's data.
/// * `input_shape` - The input's shape.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to accumulate the sum.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An i32 value representing the accumulated sum along the specified axis.
fn accumulate_sum<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut input_data: Span<T>, input_shape: Span<usize>, output_indices: Span<usize>, axis: usize
) -> T {
    let axis_len = *(input_shape)[axis];
    let mut acc: T = NumberTrait::zero();

    let mut axis_index: usize = 0;

    if (input_shape).len() > 1 {
        loop {
            if axis_index == axis_len {
                break ();
            }

            let input_indices = combine_indices(output_indices, axis_index, axis);
            let input_index = ravel_index(input_shape, input_indices);
            let ele = *(input_data)[input_index];
            acc += ele;
            axis_index += 1;
        };
    } else {
        loop {
            match input_data.pop_front() {
                Option::Some(item) => { acc += *item; },
                Option::None(_) => { break; }
            };
        };
    }

    return acc;
}
