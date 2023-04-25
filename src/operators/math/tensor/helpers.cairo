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
fn len_from_shape(shape: Span<usize>) -> usize {
    let mut result: usize = 1;

    let mut i: usize = 0;
    loop {
        check_gas();

        if i == shape.len() {
            break ();
        }

        result *= *shape.at(i);
        i += 1;
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
fn check_compatibility(shape_1: Span<usize>, shape_2: Span<usize>) {
    assert(shape_1.len() == shape_2.len(), 'tensors shape must match');

    let mut n: usize = 0;
    loop {
        check_gas();

        assert(
            *shape_1.at(
                n
            ) == *shape_2.at(n) | *shape_1.at(n) == 1_usize | *shape_2.at(n) == 1_usize,
            'tensors shape must match'
        );

        n += 1;
        if n == shape_1.len() {
            break ();
        };
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
fn broadcast_index_mapping(shape: Span<usize>, indices: Span<usize>) -> usize {
    assert(shape.len() == indices.len(), 'shape/indices len must be equal');
    let mut result = 0_usize;

    let mut n: usize = 0;
    loop {
        check_gas();

        let stride = stride(shape);
        let index = (*indices.at(n) % *shape.at(n)) * *stride.at(n);
        result += index;

        n += 1;
        if n == shape.len() {
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
fn reduce_output_shape(input_shape: Span<usize>, axis: usize) -> Span<usize> {
    assert(input_shape.len() > 0, 'input_shape cannot be empty');
    assert(axis <= input_shape.len(), 'axis is out of bound');

    let mut reduced = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        if n != axis {
            reduced.append(*input_shape.at(n));
        }

        n += 1;
        if n == input_shape.len() {
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
fn permutation_output_shape(input_shape: Span<usize>, axes: Span<usize>) -> Span<usize> {
    assert(input_shape.len() == axes.len(), 'input_shape/indices len unequal');

    let mut output_shape = ArrayTrait::new();
    let mut axis: usize = 0;

    loop {
        check_gas();
        if axis == axes.len() {
            break ();
        }

        output_shape.append(*input_shape.at(*axes.at(axis)));
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
/// * `axes` -  A span containing the usize elements representing the axes.
/// * `target_axis` - A usize representing the target axis.
///
/// # Panics
/// * Panics if the target_axis value is not in the range of the axes dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize representing the index of the target axis in the given axes array.
fn find_axis(axes: Span <usize>, target_axis: usize) -> usize {
    assert(target_axis < axes.len(), 'target_axis is out of range');

    let mut axis: usize = 0;
    loop {
        check_gas();
        if axis == axes.len() {
            break ();
        }

        if *axes.at(axis) == target_axis {
            break ();
        }
        axis += 1;
    };
    return axis;
}
