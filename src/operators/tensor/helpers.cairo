use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::utils::{check_gas, u32_max};
use orion::operators::tensor::core::stride;

/// Calculates the number of elements in a tensor given its shape.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize representing the number of elements in the tensor.
fn len_from_shape(mut shape: Span<usize>) -> usize {
    let mut result: usize = 1;

    loop {
        check_gas();

        if shape.len() == 0 {
            break ();
        }

        result *= *shape.pop_front().unwrap();
    };

    return result;
}

/// Verifies if the shape and the data array of a tensor are compatible.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
/// * `data` -  A span containing the data elements of generic type T.
///
/// # Panics
/// * Panics if the shape and data array are incompatible.
fn check_shape<T>(shape: Span<usize>, data: Span<T>) {
    assert(len_from_shape(shape) == data.len(), 'wrong tensor shape');
}

/// Checks if two tensor shapes are compatible for broadcasting.
///
/// # Arguments
/// * `shape_1` - A span containing the first tensor's shape as usize elements.
/// * `shape_2` - A span containing the second tensor's shape as usize elements.
///
/// # Panics
/// * Panics if the shapes are not compatible for broadcasting.
fn check_compatibility(mut shape_1: Span<usize>, mut shape_2: Span<usize>) {
    assert(shape_1.len() == shape_2.len(), 'tensors shape must match');

    loop {
        check_gas();

        if shape_1.len() == 0 {
            break ();
        }

        let shape_1_val = *shape_1.pop_front().unwrap();
        let shape_2_val = *shape_2.pop_front().unwrap();

        assert(
            shape_1_val == shape_2_val || shape_1_val == 1 || shape_2_val == 1,
            'tensors shape must match'
        );
    };
}

/// Computes the index in the broadcasted tensor corresponding to the given indices and shape.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
/// * `indices` - A span containing the indices as usize elements.
///
/// # Panics
/// * Panics if shape and indices length are not equal.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize representing the index in the broadcasted tensor.
fn broadcast_index_mapping(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    assert(shape.len() == indices.len(), 'shape/indices len must be equal');
    let mut result = 0_usize;
    let mut stride = stride(shape);

    loop {
        check_gas();

        let indices_val = *indices.pop_front().unwrap();
        let shape_val = *shape.pop_front().unwrap();
        let stride_val = *stride.pop_front().unwrap();

        let index = (indices_val % shape_val) * stride_val;
        result += index;

        if shape.len() == 0 {
            break ();
        };
    };

    return result;
}

/// Generates the output shape after reducing a tensor along a specified axis.
///
/// # Arguments
/// * `input_shape` - A span containing the input tensor's shape as usize elements.
/// * `axis` - A usize representing the axis to reduce.
///
/// # Panics
/// * Panics if input_shape is empty.
/// * Panic if the axis is not in the valid range of the input_shape dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the output shape after reduction.
fn reduce_output_shape(mut input_shape: Span<usize>, axis: usize, keepdims: bool) -> Span<usize> {
    assert(axis < input_shape.len(), 'axis out of dimensions');

    let mut output_shape = ArrayTrait::new();
    let mut n: usize = 0;

    loop {
        check_gas();

        if input_shape.len() == 0 {
            break ();
        }

        let current_dim = *input_shape.pop_front().unwrap();

        if n == axis {
            if keepdims {
                output_shape.append(1);
            }
        } else {
            output_shape.append(current_dim);
        }

        n += 1;
    };

    return output_shape.span();
}


/// Helper function that computes the output shape of a tensor after applying the axes permutation.
///
/// # Arguments
/// * `input_shape` - A span containing the input tensor's shape as usize elements.
/// * `axes` -  A span containing the usize elements representing the axes permutation.
///
/// # Panics
/// * Panics if shape and axes length are not equal.
/// * Panic if the axis value in axes is not in the valid range of the input_shape dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the output shape after permutation.
fn permutation_output_shape(input_shape: Span<usize>, mut axes: Span<usize>) -> Span<usize> {
    let axes_len = axes.len();
    assert(input_shape.len() == axes_len, 'input_shape/indices len unequal');

    let mut output_shape = ArrayTrait::new();
    let mut axis: usize = 0;
    loop {
        check_gas();
        if axis == axes_len {
            break ();
        }

        output_shape.append(*input_shape[*axes.pop_front().unwrap()]);
        axis += 1;
    };

    return output_shape.span();
}

/// Combines output indices with the current index of the specified axis.
///
/// # Arguments
/// * `output_indices` - A span containing the output indices as usize elements.
/// * `axis_index` - A usize representing the current index of the specified axis.
/// * `axis` - A usize representing the specified axis.
///
/// # Panics
/// * Panics if the axis value is not in the range of the output_indices length.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the combined indices.
fn combine_indices(output_indices: Span<usize>, axis_index: usize, axis: usize) -> Span<usize> {
    assert(axis <= output_indices.len(), 'axis value is out of range');

    let mut result = ArrayTrait::new();
    let output_indices_len = output_indices.len();
    let mut n: usize = 0;

    loop {
        check_gas();

        if n > output_indices_len {
            break ();
        }

        if n == axis {
            result.append(axis_index);
        } else if n > axis {
            result.append(*output_indices[n - 1_usize]);
        } else {
            result.append(*output_indices[n]);
        }

        n += 1;
    };

    return result.span();
}


/// Helper function that finds the index of a target axis in the given axes array.
///
/// # Arguments
/// * `axes` - A span containing the usize elements representing the axes.
/// * `target_axis` - A usize representing the target axis.
///
/// # Panics
/// * Panics if the target_axis value is not in the range of the axes dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize representing the index of the target axis in the given axes array.
fn find_axis(mut axes: Span<usize>, target_axis: usize) -> usize {
    assert(target_axis < axes.len(), 'target_axis is out of range');

    let mut axis: usize = 0;
    loop {
        check_gas();

        if axes.len() == 0 {
            break ();
        }

        let current_axis = *axes.pop_front().unwrap();
        if current_axis == target_axis {
            break ();
        }
        axis += 1;
    };
    return axis;
}

/// Computes the broadcasted shape of two tensors.
///
/// # Arguments
/// * `shape1` - A span containing the shape of the first tensor as usize elements.
/// * `shape2` - A span containing the shape of the second tensor as usize elements.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the broadcasted shape.
fn broadcast_shape(mut shape1: Span<usize>, mut shape2: Span<usize>) -> Span<usize> {
    check_compatibility(shape1, shape2);
    let mut result: Array<usize> = ArrayTrait::new();
    let mut temp_result = ArrayTrait::new();

    loop {
        check_gas();

        // Get dimensions from shape1 and shape2, or use 1 if there are no more dimensions
        let dim1 = if shape1.len() > 0 {
            *shape1.pop_back().unwrap()
        } else {
            1
        };

        let dim2 = if shape2.len() > 0 {
            *shape2.pop_back().unwrap()
        } else {
            1
        };

        let broadcasted_dim = u32_max(dim1, dim2);
        temp_result.append(broadcasted_dim);

        if shape1.len() == 0 && shape2.len() == 0 {
            break ();
        };
    };

    // Copy the broadcasted dimensions to the result array in the correct order
    let mut temp_result: Span<usize> = temp_result.span();
    loop {
        check_gas();

        if temp_result.len() == 0 {
            break ();
        }
        result.append(*temp_result.pop_back().unwrap());
    };

    return result.span();
}


/// Substitute a value in a shape at a given index
/// 
/// # Arguments
///
/// * `shape` - The shape to modify
/// * `index` - The index to modify
/// * `value` - The value to insert
///
/// # Panics
/// * Panics if the index is out of bounds
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * `Span<usize>` - The modified shape
fn replace_index(mut shape: Span<usize>, index: usize, value: usize) -> Span<usize> {
    let mut output = ArrayTrait::new();
    let mut i = 0;
    loop {
        if i == shape.len() {
            break ();
        };
        if i == index {
            output.append(value);
        } else {
            output.append(*shape[i]);
        };
        i += 1;
    };
    return output.span();
}
