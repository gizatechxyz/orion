//! This module defines and implement a Tensor for i32 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::helpers::check_shape;
use onnx_cairo::operators::tensor::helpers::check_compatibility;
use onnx_cairo::operators::tensor::helpers::broadcast_shape;
use onnx_cairo::operators::tensor::core::new_tensor;
use onnx_cairo::operators::tensor::core::stride;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::core::ravel_index;
use onnx_cairo::operators::tensor::core::unravel_index;
use onnx_cairo::operators::tensor::core::reshape;
use onnx_cairo::operators::tensor::helpers::broadcast_index_mapping;
use onnx_cairo::operators::tensor::helpers::reduce_output_shape;
use onnx_cairo::operators::tensor::helpers::len_from_shape;
use onnx_cairo::operators::tensor::helpers::combine_indices;
use onnx_cairo::operators::tensor::helpers::find_axis;
use onnx_cairo::operators::tensor::helpers::permutation_output_shape;
use onnx_cairo::operators::tensor::tensor_u32;
use onnx_cairo::operators::math::min::min_i32::min_in_tensor;
use onnx_cairo::operators::math::max::max_i32::max_in_tensor;
use onnx_cairo::operators::math::reduce_sum::reduce_sum_i32::reduce_sum;
use onnx_cairo::operators::math::argmax::argmax_i32::argmax;
use onnx_cairo::operators::linalg::matmul::matmul_i32::matmul;
use onnx_cairo::utils::check_gas;

impl i32Tensor of TensorTrait<i32> {
    /// Creates a new i32 tensor with the given shape and data.
    ///
    /// # Arguments
    /// * `shape` - A span representing the shape of the tensor.
    /// * `data` -  A span containing the array of i32 elements.
    ///
    /// # Panics
    /// * Panics if the shape and data length are incompatible.
    ///
    /// # Returns
    /// * A new `Tensor<i32>` instance.
    fn new(shape: Span<usize>, data: Span<i32>) -> Tensor<i32> {
        new_tensor(shape, data)
    }

    /// Retrieves the value at the specified indices of an i32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `indices` - A span representing the indices to access.
    ///
    /// # Panics
    /// * Panics the number of indices provided don't match the number of dimensions in the tensor.
    ///
    /// # Returns
    /// * The i32 value at the specified indices.
    fn at(self: @Tensor<i32>, indices: Span<usize>) -> i32 {
        i32_at_tensor(self, indices)
    }

    /// Finds the minimum value in an i32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The minimum i32 value in the tensor.
    fn min(self: @Tensor<i32>) -> i32 {
        min_in_tensor(*self.data)
    }

    /// Finds the maximum value in an i32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The maximum i32 value in the tensor.
    fn max(self: @Tensor<i32>) -> i32 {
        max_in_tensor(*self.data)
    }

    /// Computes the stride of an i32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if shape is empty.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A span representing the stride of the tensor.
    fn stride(self: @Tensor<i32>) -> Span<usize> {
        stride(*self.shape)
    }

    /// Converts a multi-dimensional index to a one-dimensional index.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `indices` - A span representing the indices.
    ///
    /// # Panics
    /// * Panics if the indices are out of bounds for the given shape.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The raveled index corresponding to the given indices.
    fn ravel_index(self: @Tensor<i32>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    /// Converts a one-dimensional index to a multi-dimensional index.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `index` - The index to unravel.
    ///
    /// # Panics
    /// * Panics if the index is out of bounds for the given shape.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A span representing the unraveled indices corresponding to the given index.
    fn unravel_index(self: @Tensor<i32>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    /// Reshapes an i32 tensor to the target shape.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `target_shape` - A span representing the target shape.
    ///
    /// # Panics
    /// * Panics if the target shape is incompatible with the input tensor's data.
    ///
    /// # Returns
    /// * A new `Tensor<i32>` instance with the specified shape.
    fn reshape(self: @Tensor<i32>, target_shape: Span<usize>) -> Tensor<i32> {
        reshape(self, target_shape)
    }

    /// Reduces an i32 tensor along the given axis by summing its elements.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `axis` - The axis along which to reduce the tensor.
    ///
    /// # Panics
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new `Tensor<i32>` instance with the specified axis reduced by summing its elements.
    fn reduce_sum(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
        reduce_sum(self, axis)
    }

    /// Computes the indices of the maximum values along the given axis of an i32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `axis` - The axis along which to compute the argmax.
    ///
    /// # Panics
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new `Tensor<usize>` instance containing the indices of the maximum values along the specified axis.
    fn argmax(self: @Tensor<i32>, axis: usize) -> Tensor<usize> {
        argmax(self, axis)
    }

    /// Transposes an i32 tensor according to the specified axes.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `axes` -  A span containing the array representing the order in which the axes should be transposed.
    ///
    /// # Panics
    /// * Panics if the length of the axes array is not equal to the rank of the input tensor.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new `Tensor<i32>` instance with the axes transposed according to the specified order.
    fn transpose(self: @Tensor<i32>, axes: Span<usize>) -> Tensor<i32> {
        i32_transpose(self, axes)
    }

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
        matmul(self, other)
    }
}

/// Implements addition for `Tensor<i32>` using the `Add` trait.
impl i32TensorAdd of Add<Tensor<i32>> {
    /// Adds two `Tensor<i32>` instances element-wise.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i32>` instance representing the result of the element-wise addition.
    fn add(self: Tensor<i32>, other: Tensor<i32>) -> Tensor<i32> {
        i32_add_tensor(@self, @other)
    }
}

/// Implements subtraction for `Tensor<i32>` using the `Sub` trait.
impl i32TensorSub of Sub<Tensor<i32>> {
    /// Subtracts two `Tensor<i32>` instances element-wise.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i32>` instance representing the result of the element-wise subtraction.
    fn sub(self: Tensor<i32>, other: Tensor<i32>) -> Tensor<i32> {
        i32_sub_tensor(@self, @other)
    }
}

/// Implements multiplication for `Tensor<i32>` using the `Mul` trait.
impl i32TensorMul of Mul<Tensor<i32>> {
    /// Multiplies two `Tensor<i32>` instances element-wise.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i32>` instance representing the result of the element-wise multiplication.
    fn mul(self: Tensor<i32>, other: Tensor<i32>) -> Tensor<i32> {
        i32_mul_tensor(@self, @other)
    }
}

/// Implements division for `Tensor<i32>` using the `Div` trait.
impl i32TensorDiv of Div<Tensor<i32>> {
    /// Divides two `Tensor<i32>` instances element-wise.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i32>` instance representing the result of the element-wise division.
    fn div(self: Tensor<i32>, other: Tensor<i32>) -> Tensor<i32> {
        i32_div_tensor(@self, @other)
    }
}

/// Retrieves the value at the specified indices in a `Tensor<i32>`.
///
/// # Arguments
/// * `self` - The tensor.
/// * `indices` - A span containing the indices as usize elements.
///
/// # Panics
/// * Panics the number of indices provided don't match the number of dimensions in the tensor.
///
/// # Returns
/// * An i32 value at the specified indices in the tensor.
fn i32_at_tensor(self: @Tensor<i32>, indices: Span<usize>) -> i32 {
    assert(indices.len() == (*self.shape).len(), 'indices not match dimensions');
    let data = *self.data;
    *data.at(self.ravel_index(indices))
}

// --- BROADCAST OPERATIONS ---

/// Adds two `Tensor<i32>` instances element-wise with broadcasting.
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
/// * A `Tensor<i32>` instance representing the result of the element-wise addition with broadcasting.
fn i32_add_tensor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
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

    return TensorTrait::<i32>::new(broadcasted_shape, result.span());
}

/// Subtracts two `Tensor<i32>` instances element-wise with broadcasting.
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
/// * A `Tensor<i32>` instance representing the result of the element-wise subtraction with broadcasting.
fn i32_sub_tensor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
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

    return TensorTrait::<i32>::new(broadcasted_shape, result.span());
}

/// Multiplies two `Tensor<i32>` instances element-wise with broadcasting.
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
/// * A `Tensor<i32>` instance representing the result of the element-wise multiplication with broadcasting.
fn i32_mul_tensor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
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

    return TensorTrait::<i32>::new(broadcasted_shape, result.span());
}

/// Divides two `Tensor<i32>` instances element-wise with broadcasting.
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
/// * A `Tensor<i32>` instance representing the result of the element-wise division with broadcasting.
fn i32_div_tensor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
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

    return TensorTrait::<i32>::new(broadcasted_shape, result.span());
}

/// Reorders the axes of an i32 tensor according to the given axes permutation.
///
/// # Arguments
/// * `self` - The input tensor.
/// * `axes` -  A span containing the usize elements representing the axes permutation.
///
/// # Panics
/// * Panics if the length of the axes array is not equal to the rank of the input tensor.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<i32>` instance with the axes reordered according to the given permutation.
fn i32_transpose(self: @Tensor<i32>, axes: Span<usize>) -> Tensor<i32> {
    assert(axes.len() == (*self.shape).len(), 'shape and axes length unequal');

    let output_shape = permutation_output_shape(*self.shape, axes);
    let output_data_len = len_from_shape(output_shape);

    let mut output_data = ArrayTrait::new();

    let mut output_index: usize = 0;
    loop {
        check_gas();

        if output_index == output_data_len {
            break ();
        }

        let output_indices = unravel_index(output_index, output_shape);
        let mut input_indices = ArrayTrait::new();

        let mut output_axis: usize = 0;
        loop {
            check_gas();
            if output_axis == axes.len() {
                break ();
            }

            let input_axis = find_axis(axes, output_axis);
            input_indices.append(*output_indices.at(input_axis));
            output_axis += 1;
        };

        let input_index = ravel_index(*self.shape, input_indices.span());
        output_data.append(*(*self.data).at(input_index));

        output_index += 1;
    };

    return TensorTrait::<i32>::new(output_shape, output_data.span());
}
