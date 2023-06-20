use core::debug::PrintTrait;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::implementations:: impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape,combine_indices};
use orion::utils::check_gas;

/// Helper function that finds the index of the minimum value in a flat tensor.
///
/// # Arguments
/// * `input` - The input tensor.
/// * `axis` - The axis along which to find the minimum value.
/// * `keepdims` - Whether to keep the reduced dimension or not.
/// * `select_last_index` - Whether to select last occurrence of the min value along the axis.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize value representing the index of the minimum value along the specified axis.
fn find_argmin_1D< T, 
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    input: @Tensor<T>,
    axis: usize, 
    keepdims:bool, 
    select_last_index: bool
    ) -> Tensor<usize>{

        let mut output_data = ArrayTrait::<usize>::new();
        let mut data = *input.data;

        let mut min = *data.pop_front().unwrap();
        let mut min_index = 0_usize;
        let mut count = 0_usize;
        loop {
            check_gas();
            
            if data.len() == 0 {
                break ();
            };

            count += 1;

            let current_value = *data.pop_front().unwrap();
            if current_value < min {
                min = current_value;
                min_index = count;

            } else {
                if select_last_index & (current_value == min) {
                    min_index = count;
                }
            }; 
        };

        output_data.append(min_index);

        return TensorTrait::<usize>::new(reduce_output_shape(*input.shape, axis, keepdims), output_data.span(), *input.extra);
}



/// Recursive helper function that finds the index of the minimum value along a specific axis.
///
/// # Arguments
/// * `input` - The input tensor.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to find the minimum value.
/// * `axis_index` - The current index along the specified axis.
/// * `min_value` - The current minimum value found along the axis.
/// * `argmin` - The current index of the minimum value along the axis.
/// * `select_last_index` - Whether to select last occurrence of the min value along the axis.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize value representing the index of the minimum value along the specified axis.
fn find_argmin< T, 
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    input: @Tensor<T>,
    output_indices: Span<usize>,
    axis: usize,
    axis_index: usize,
    min_value: T,
    argmin: usize,
    select_last_index: bool
) -> usize {
    check_gas();

    if axis_index == *(*input.shape).at(axis) {
        return argmin;
    }

    let input_indices = combine_indices(output_indices, axis_index, axis);
    let input_index = ravel_index(*input.shape, input_indices);
    let ele = *(*input.data).at(input_index);

    let (new_min_value, new_argmin) = if ele < min_value {
        (ele, axis_index)
    } else {
        if select_last_index & (ele == min_value) {
            (ele, axis_index)
        } else {
            (min_value, argmin)
        }
    };

    return find_argmin(
        input, output_indices, axis, axis_index + 1_usize, new_min_value, new_argmin,select_last_index
    );
}

