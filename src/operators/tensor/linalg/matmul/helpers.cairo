use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;

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
fn prepare_shape_for_matmul(mut shape: Span<usize>, is_first_tensor: bool) -> Span<usize> {
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
fn adjust_output_shape_after_matmul(
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
