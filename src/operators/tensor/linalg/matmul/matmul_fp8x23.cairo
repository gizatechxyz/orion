use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::utils::check_gas;
use orion::operators::tensor::implementations::impl_tensor_fp8x23;
use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::linalg::matmul::helpers::{
    prepare_shape_for_matmul, adjust_output_shape_after_matmul
};

/// Cf: TensorTrait::matmul docstring
fn matmul(
    self: @Tensor<FixedType<fp8x23>>, other: @Tensor<FixedType<fp8x23>>
) -> Tensor<FixedType<fp8x23>> {
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
        return TensorTrait::<FixedType<fp8x23>,
        fp8x23>::new(result_shape.span(), result_data.span());
    }

    let self_shape = prepare_shape_for_matmul(self_shape, true);
    let other_shape = prepare_shape_for_matmul(other_shape, false);

    let result = matrix_multiply(*self.data, self_shape, *other.data, other_shape);

    let result_shape = adjust_output_shape_after_matmul(result.shape, self_ndim, other_ndim);

    return TensorTrait::<FixedType>::new(result_shape, result.data);
}

/// Computes the dot product of two 1-dimensional FixedType tensors.
///
/// # Arguments
/// * `vec1` - A span containing the data elements of the first vector as FixedType elements.
/// * `vec2` - A span containing the data elements of the second vector as FixedType elements.
///
/// # Panics
/// * Panics if the lengths of the vectors do not match.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An FixedType representing the dot product of the two vectors.
fn dot_product(
    mut vec1: Span<FixedType<fp8x23>>, mut vec2: Span<FixedType<fp8x23>>
) -> FixedType<fp8x23> {
    assert(vec1.len() == vec2.len(), 'vector lengths do not match');

    let mut result: FixedType = FixedTrait::new_unscaled(0, false);

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


/// Computes the matrix multiplication of two 2-dimensional FixedType tensors.
///
/// # Arguments
/// * `mat1` - A Span containing the data elements of the first matrix as FixedType elements.
/// * `mat1_shape` - A Span containing the shape of the first matrix as usize elements.
/// * `mat2` - A Span containing the data elements of the second matrix as FixedType elements.
/// * `mat2_shape` - A Span containing the shape of the second matrix as usize elements.
///
/// # Panics
/// * Panics if the inner dimensions of the matrices do not match.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * Returns the restulting FixedType tensor.
fn matrix_multiply(
    mat1: Span<FixedType<fp8x23>>,
    mat1_shape: Span<usize>,
    mat2: Span<FixedType<fp8x23>>,
    mat2_shape: Span<usize>
) -> Tensor<FixedType<fp8x23>> {
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

            let mut sum: FixedType = FixedTrait::<fp8x23>::new_unscaled(0, false);
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

    return TensorTrait::<FixedType<fp8x23>, fp8x23>::new(result_shape.span(), result_data.span());
}