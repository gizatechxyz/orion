use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::operators::tensor::helpers::broadcast_shape;

use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait, unravel_index, };
use onnx_cairo::operators::tensor::helpers::{broadcast_index_mapping, len_from_shape, };
use onnx_cairo::operators::tensor::implementations::impl_tensor_u32;
use onnx_cairo::utils::check_gas;

/// Adds two `Tensor<u32>` instances element-wise with broadcasting.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
///
/// # Panics
/// * Panics if the shape of tensors are not compatible. 
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the element-wise addition with broadcasting.
fn add(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data).at(indices_self) + *(*other.data).at(indices_other));

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<u32>::new(broadcasted_shape, result.span());
}

/// Subtracts two `Tensor<u32>` instances element-wise with broadcasting.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
///
/// # Panics
/// * Panics if the shape of tensors are not compatible. 
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the element-wise subtraction with broadcasting.
fn sub(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data).at(indices_self) - *(*other.data).at(indices_other));

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<u32>::new(broadcasted_shape, result.span());
}

/// Multiplies two `Tensor<u32>` instances element-wise with broadcasting.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
///
/// # Panics
/// * Panics if the shape of tensors are not compatible. 
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the element-wise multiplication with broadcasting.
fn mul(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data).at(indices_self) * *(*other.data).at(indices_other));

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<u32>::new(broadcasted_shape, result.span());
}

/// Divides two `Tensor<u32>` instances element-wise with broadcasting.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
///
/// # Panics
/// * Panics if the shape of tensors are not compatible. 
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the element-wise division with broadcasting.
fn div(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data).at(indices_self) / *(*other.data).at(indices_other));

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<u32>::new(broadcasted_shape, result.span());
}