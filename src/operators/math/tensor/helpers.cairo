use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::math::tensor::core::stride;

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
            shape_1_val == shape_2_val | shape_1_val == 1 | shape_2_val == 1,
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
fn reduce_output_shape(mut input_shape: Span<usize>, axis: usize) -> Span<usize> {
    let input_shape_len = input_shape.len();

    assert(input_shape_len > 0, 'input_shape cannot be empty');
    assert(axis <= input_shape_len, 'axis is out of bound');

    let mut reduced = ArrayTrait::new();
    let mut current_axis: usize = 0;
    loop {
        check_gas();

        if current_axis != axis {
            reduced.append(*input_shape.pop_front().unwrap());
        }

        current_axis += 1;
        if current_axis == input_shape_len {
            break ();
        };
    };

    return reduced.span();
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

        output_shape.append(*input_shape.at(*axes.pop_front().unwrap()));
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
            result.append(*output_indices.at(n - 1_usize));
        } else {
            result.append(*output_indices.at(n));
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

/// Prepares the shape of a tensor for matrix multiplication.
///
/// # Arguments
/// * `shape` - A mutable span representing the shape of the tensor.
/// * `is_first_tensor` - A boolean indicating whether the input tensor is the first (left) 
///   tensor in the matrix multiplication operation.
///
/// # Behavior
/// This function adjusts the shapes of the tensors based on their dimensionality:
/// * If the first tensor is 1-dimensional, a 1 is prepended to its shape.
/// * If the second tensor is 1-dimensional, a 1 is appended to its shape.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A span representing the adjusted shape of the tensor.
fn prepare_shape_for_mul(mut shape: Span<usize>, is_first_tensor: bool) -> Span<usize> {
    let ndim = shape.len();

    if ndim == 1 & is_first_tensor {
        // Prepend 1 to shape if it's 1-dimensional
        let mut shape_adjusted = ArrayTrait::new();
        shape_adjusted.append(1);
        loop {
            check_gas();
            if shape.len() == 0 {
                break ();
            }
            shape_adjusted.append(*shape.pop_front().unwrap());
        };

        return shape_adjusted.span();
    } else if ndim == 1 & !is_first_tensor {
        // Append 1 to shape if it's 1-dimensional
        let mut shape_adjusted = ArrayTrait::new();
        loop {
            check_gas();
            if shape.len() == 0 {
                break ();
            }
            shape_adjusted.append(*shape.pop_front().unwrap());
        };
        shape_adjusted.append(1);

        return shape_adjusted.span();
    }

    return shape;
}

/// Adjusts the output shape of the matrix multiplication result based on the
/// original dimensionality of the input tensors.
///
/// # Arguments
/// * `output_shape` - A mutable span representing the shape of the matrix multiplication result.
/// * `self_dim` - A usize representing the dimensionality of the first input tensor.
/// * `other_dim` - A usize representing the dimensionality of the second input tensor.
///
/// # Behavior
/// This function adjusts the output shape based on the dimensionality of the input tensors:
/// * If the first input tensor was 1-dimensional, the prepended 1 is removed from the output shape.
/// * If the second input tensor was 1-dimensional, the appended 1 is removed from the output shape.
///
/// # Returns
/// * A span representing the adjusted output shape of the matrix multiplication result.
fn adjust_output_shape_after_mul(
    mut output_shape: Span<usize>, self_dim: usize, other_dim: usize
) -> Span<usize> {
    // If self_shape was 1-dimensional, remove the prepended 1 from the output_shape.
    if self_dim == 1 {
        let _ = output_shape.pop_front().unwrap();
    }

    // If other_shape was 1-dimensional, remove the appended 1 from the output_shape.
    if other_dim == 1 {
        let _ = output_shape.pop_back().unwrap();
    }

    return output_shape;
}
