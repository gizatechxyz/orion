use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::matmul docstring
fn matmul<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, other: @Tensor<T>
) -> Tensor<T> {
    let self_shape = *self.shape;
    let other_shape = *other.shape;
    let self_ndim = (self_shape).len();
    let other_ndim = (other_shape).len();

    assert(self_ndim <= 2 || other_ndim <= 2, 'supports only 1D and 2D matmul');

    //! Case: Both tensors are 1-dimensional
    if self_ndim == 1 && other_ndim == 1 {
        let dot = dot_product((*self).data, (*other).data);
        let mut result_shape = ArrayTrait::new();
        let mut result_data = ArrayTrait::new();
        result_shape.append(1);
        result_data.append(dot);
        return TensorTrait::new(result_shape.span(), result_data.span());
    }

    let self_shape = prepare_shape_for_matmul(self_shape, true);
    let other_shape = prepare_shape_for_matmul(other_shape, false);

    let result = matrix_multiply(*self.data, self_shape, *other.data, other_shape);

    let result_shape = adjust_output_shape_after_matmul(result.shape, self_ndim, other_ndim);

    return TensorTrait::new(result_shape, result.data);
}

/// Computes the dot product of two 1-dimensional i32 tensors.
///
/// # Arguments
/// * `vec1` - A span containing the data elements of the first vector as i32 elements.
/// * `vec2` - A span containing the data elements of the second vector as i32 elements.
///
/// # Panics
/// * Panics if the lengths of the vectors do not match.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An i32 representing the dot product of the two vectors.
fn dot_product<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut vec1: Span<T>, mut vec2: Span<T>
) -> T {
    assert(vec1.len() == vec2.len(), 'vector lengths do not match');

    let mut result: T = NumberTrait::zero();

    loop {
        match vec1.pop_front() {
            Option::Some(vec1_item) => {
                let element_product = *vec1_item * *vec2.pop_front().unwrap();
                result += element_product;
            },
            Option::None => { break; }
        };
    };

    return result;
}


/// Computes the matrix multiplication of two 2-dimensional i32 tensors.
///
/// # Arguments
/// * `mat1` - A Span containing the data elements of the first matrix as i32 elements.
/// * `mat1_shape` - A Span containing the shape of the first matrix as usize elements.
/// * `mat2` - A Span containing the data elements of the second matrix as i32 elements.
/// * `mat2_shape` - A Span containing the shape of the second matrix as usize elements.
///
/// # Panics
/// * Panics if the inner dimensions of the matrices do not match.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * Returns the restulting i32 tensor.
fn matrix_multiply<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mat1: Span<T>, mat1_shape: Span<usize>, mat2: Span<T>, mat2_shape: Span<usize>
) -> Tensor<T> {
    let m = *mat1_shape[0];
    let n = *mat1_shape[1];
    let p = *mat2_shape[1];

    let mut result_data = ArrayTrait::new();
    let mut result_shape = ArrayTrait::new();
    result_shape.append(m);
    result_shape.append(p);

    let mut i = 0_usize;
    loop {
        if i == m {
            break ();
        }

        let mut j = 0_usize;
        loop {
            if j == p {
                break ();
            }

            let mut sum: T = NumberTrait::zero();
            let mut k = 0_usize;
            loop {
                if k == n {
                    break ();
                }

                let mat1_index = i * n + k;
                let mat2_index = k * p + j;
                sum += *mat1[mat1_index] * *mat2[mat2_index];

                k += 1;
            };

            result_data.append(sum);
            j += 1;
        };

        i += 1;
    };

    return TensorTrait::new(result_shape.span(), result_data.span());
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
fn prepare_shape_for_matmul(mut shape: Span<usize>, is_first_tensor: bool) -> Span<usize> {
    let ndim = shape.len();

    if ndim == 1 && is_first_tensor {
        // Prepend 1 to shape if it's 1-dimensional
        let mut shape_adjusted = ArrayTrait::new();
        shape_adjusted.append(1);

        loop {
            match shape.pop_front() {
                Option::Some(item) => { shape_adjusted.append(*item); },
                Option::None => { break; }
            };
        };

        return shape_adjusted.span();
    } else if ndim == 1 && !is_first_tensor {
        // Append 1 to shape if it's 1-dimensional
        let mut shape_adjusted = ArrayTrait::new();

        loop {
            match shape.pop_front() {
                Option::Some(item) => { shape_adjusted.append(*item); },
                Option::None => { break; }
            };
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
