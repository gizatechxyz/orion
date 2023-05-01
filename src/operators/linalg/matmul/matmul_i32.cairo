use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::tensor_i32;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::core::ravel_index;
use onnx_cairo::operators::tensor::core::unravel_index;
use onnx_cairo::operators::tensor::helpers::reduce_output_shape;
use onnx_cairo::operators::tensor::helpers::len_from_shape;
use onnx_cairo::operators::tensor::helpers::combine_indices;
use onnx_cairo::operators::linalg::matmul::helpers::prepare_shape_for_matmul;
use onnx_cairo::operators::linalg::matmul::helpers::adjust_output_shape_after_matmul;
use onnx_cairo::utils::check_gas;

/// Performs matrix multiplication between two i32 tensors.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
///
/// # Behavior
/// The behavior depends on the dimensionality of the tensors as follows:
/// * If both tensors are 1-dimensional, the dot product is returned.
/// * If both arguments are 2-dimensional, the matrix-matrix product is returned.
/// * If the first argument is 1-dimensional and the second argument is 2-dimensional,
///   a 1 is prepended to its dimension for the purpose of the matrix multiply. After
///   the matrix multiply, the prepended dimension is removed.
/// * If the first argument is 2-dimensional and the second argument is 1-dimensional,
///   the matrix-vector product is returned.
///
/// # Panics
/// * Panics if the dimension of the tensors is higher than two.
///
/// # Returns
/// * A new `Tensor<i32>` resulting from the matrix multiplication.
fn matmul(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
    let self_shape = *self.shape;
    let other_shape = *other.shape;
    let self_ndim = (self_shape).len();
    let other_ndim = (other_shape).len();

    assert(self_ndim <= 2 | other_ndim <= 2, 'supports only 1D and 2D matmul');

    //! Case: Both tensors are 1-dimensional
    if self_ndim == 1 & other_ndim == 1 {
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

    return TensorTrait::<i32>::new(result_shape, result.data);
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
fn dot_product(mut vec1: Span<i32>, mut vec2: Span<i32>) -> i32 {
    assert(vec1.len() == vec2.len(), 'vector lengths do not match');

    let mut result: i32 = IntegerTrait::new(0, false);
    let vec_len = vec1.len();
    let mut idx: usize = 0;

    loop {
        check_gas();
        if vec1.len() == 0 {
            break ();
        }

        let element_product = *vec1.pop_front().unwrap() * *vec2.pop_front().unwrap();
        result += element_product;
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
fn matrix_multiply(
    mat1: Span<i32>, mat1_shape: Span<usize>, mat2: Span<i32>, mat2_shape: Span<usize>
) -> Tensor<i32> {
    let m = *mat1_shape.at(0);
    let n = *mat1_shape.at(1);
    let p = *mat2_shape.at(1);

    let mut result_data = ArrayTrait::new();
    let mut result_shape = ArrayTrait::new();
    result_shape.append(m);
    result_shape.append(p);

    let mut i = 0_usize;
    loop {
        check_gas();
        if i == m {
            break ();
        }

        let mut j = 0_usize;
        loop {
            check_gas();
            if j == p {
                break ();
            }

            let mut sum: i32 = IntegerTrait::new(0, false);
            let mut k = 0_usize;
            loop {
                check_gas();
                if k == n {
                    break ();
                }

                let mat1_index = i * n + k;
                let mat2_index = k * p + j;
                sum += *mat1.at(mat1_index) * *mat2.at(mat2_index);

                k += 1;
            };

            result_data.append(sum);
            j += 1;
        };

        i += 1;
    };

    return TensorTrait::new(result_shape.span(), result_data.span());
}
