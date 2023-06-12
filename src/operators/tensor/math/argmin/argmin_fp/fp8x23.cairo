use array::ArrayTrait;
use array::SpanTrait;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23;

use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::utils::check_gas;


/// Cf: TensorTrait::argmin docstring
fn argmin(
    self: @Tensor<FixedType>, 
    axis: usize, 
    keepdims: Option<bool>, 
    select_last_index:Option<bool> 
    ) -> Tensor<usize> {

    let keepdims = match keepdims {
        Option::Some(val) => val,
        Option::None(_) => true,
    };

    let select_last_index = match select_last_index {
        Option::Some(val) => val,
        Option::None(_) => false,
    };

    assert(axis <= (*self.shape).len(), 'axis out of dimensions');

    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_output_shape(*self.shape, axis, false);
    let output_data_len = len_from_shape(output_shape);
    
    let MAX = FixedTrait::new(impl_8x23::MAX , false);

    let mut index: usize = 0;
    loop {
        check_gas();

        let output_indices = unravel_index(index, output_shape);
        let current_argmin = find_argmin(
            self, output_indices, axis, 0, MAX, 0, select_last_index
        );

        output_data.append(current_argmin);

        index += 1;
        if index == output_data_len {
            break ();
        };
    };

    return TensorTrait::<usize>::new(reduce_output_shape(*self.shape, axis, keepdims), output_data.span(), *self.extra);
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
fn find_argmin(
    input: @Tensor<FixedType>,
    output_indices: Span<usize>,
    axis: usize,
    axis_index: usize,
    min_value: FixedType,
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
