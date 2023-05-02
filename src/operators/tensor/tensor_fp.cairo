//! This module defines and implement a Tensor for FixedType values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::fixed_point::types::Fixed;
use onnx_cairo::operators::math::fixed_point::types::FixedType;
use onnx_cairo::operators::tensor::helpers::check_shape;
use onnx_cairo::operators::tensor::helpers::check_compatibility;
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
use onnx_cairo::operators::math::min::min_fp::min_in_tensor;
use onnx_cairo::operators::math::max::max_fp::max_in_tensor;
use onnx_cairo::operators::math::reduce_sum::reduce_sum_fp::reduce_sum;
use onnx_cairo::operators::math::argmax::argmax_fp::argmax;
use onnx_cairo::operators::linalg::matmul::matmul_fp::matmul;
use onnx_cairo::utils::check_gas;

impl FixedTypeTensor of TensorTrait<FixedType> {
    /// Creates a new FixedType tensor with the given shape and data.
    ///
    /// # Arguments
    /// * `shape` - A span representing the shape of the tensor.
    /// * `data` -  A span containing the array of FixedType elements.
    ///
    /// # Panics
    /// * Panics if the shape and data length are incompatible.
    ///
    /// # Returns
    /// * A new `Tensor<FixedType>` instance.
    fn new(shape: Span<usize>, data: Span<FixedType>) -> Tensor<FixedType> {
        new_tensor(shape, data)
    }

    /// Retrieves the value at the specified indices of an FixedType tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `indices` - A span representing the indices to access.
    ///
    /// # Panics
    /// * Panics the number of indices provided don't match the number of dimensions in the tensor.
    ///
    /// # Returns
    /// * The FixedType value at the specified indices.
    fn at(self: @Tensor<FixedType>, indices: Span<usize>) -> FixedType {
        FixedType_at_tensor(self, indices)
    }

    /// Finds the minimum value in an FixedType tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The minimum FixedType value in the tensor.
    fn min(self: @Tensor<FixedType>) -> FixedType {
        min_in_tensor(*self.data)
    }

    /// Finds the maximum value in an FixedType tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The maximum FixedType value in the tensor.
    fn max(self: @Tensor<FixedType>) -> FixedType {
        max_in_tensor(*self.data)
    }

    /// Computes the stride of an FixedType tensor.
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
    fn stride(self: @Tensor<FixedType>) -> Span<usize> {
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
    fn ravel_index(self: @Tensor<FixedType>, indices: Span<usize>) -> usize {
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
    fn unravel_index(self: @Tensor<FixedType>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    /// Reshapes an FixedType tensor to the target shape.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `target_shape` - A span representing the target shape.
    ///
    /// # Panics
    /// * Panics if the target shape is incompatible with the input tensor's data.
    ///
    /// # Returns
    /// * A new `Tensor<FixedType>` instance with the specified shape.
    fn reshape(self: @Tensor<FixedType>, target_shape: Span<usize>) -> Tensor<FixedType> {
        reshape(self, target_shape)
    }

    /// Reduces an FixedType tensor along the given axis by summing its elements.
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
    /// * A new `Tensor<FixedType>` instance with the specified axis reduced by summing its elements.
    fn reduce_sum(self: @Tensor<FixedType>, axis: usize) -> Tensor<FixedType> {
        reduce_sum(self, axis)
    }

    /// Computes the indices of the maximum values along the given axis of an FixedType tensor.
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
    fn argmax(self: @Tensor<FixedType>, axis: usize) -> Tensor<usize> {
        argmax(self, axis)
    }

    /// Transposes an FixedType tensor according to the specified axes.
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
    /// * A new `Tensor<FixedType>` instance with the axes transposed according to the specified order.
    fn transpose(self: @Tensor<FixedType>, axes: Span<usize>) -> Tensor<FixedType> {
        FixedType_transpose(self, axes)
    }

    /// Performs matrix multiplication between two FixedType tensors.
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
    /// * A new `Tensor<FixedType>` resulting from the matrix multiplication.
    fn matmul(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
        matmul(self, other)
    }
}

/// Implements addition for `Tensor<FixedType>` using the `Add` trait.
impl FixedTypeTensorAdd of Add<Tensor<FixedType>> {
    /// Adds two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise addition.
    fn add(self: Tensor<FixedType>, other: Tensor<FixedType>) -> Tensor<FixedType> {
        FixedType_add_tensor(@self, @other)
    }
}

/// Implements subtraction for `Tensor<FixedType>` using the `Sub` trait.
impl FixedTypeTensorSub of Sub<Tensor<FixedType>> {
    /// Subtracts two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise subtraction.
    fn sub(self: Tensor<FixedType>, other: Tensor<FixedType>) -> Tensor<FixedType> {
        FixedType_sub_tensor(@self, @other)
    }
}

/// Implements multiplication for `Tensor<FixedType>` using the `Mul` trait.
impl FixedTypeTensorMul of Mul<Tensor<FixedType>> {
    /// Multiplies two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise multiplication.
    fn mul(self: Tensor<FixedType>, other: Tensor<FixedType>) -> Tensor<FixedType> {
        FixedType_mul_tensor(@self, @other)
    }
}

/// Implements division for `Tensor<FixedType>` using the `Div` trait.
impl FixedTypeTensorDiv of Div<Tensor<FixedType>> {
    /// Divides two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise division.
    fn div(self: Tensor<FixedType>, other: Tensor<FixedType>) -> Tensor<FixedType> {
        FixedType_div_tensor(@self, @other)
    }
}

/// Retrieves the value at the specified indices in a `Tensor<FixedType>`.
///
/// # Arguments
/// * `self` - The tensor.
/// * `indices` - A span containing the indices as usize elements.
///
/// # Panics
/// * Panics the number of indices provided don't match the number of dimensions in the tensor.
///
/// # Returns
/// * An FixedType value at the specified indices in the tensor.
fn FixedType_at_tensor(self: @Tensor<FixedType>, indices: Span<usize>) -> FixedType {
    assert(indices.len() == (*self.shape).len(), 'indices not match dimensions');
    let data = *self.data;
    *data.at(self.ravel_index(indices))
}

// --- BROADCAST OPERATIONS ---

/// Adds two `Tensor<FixedType>` instances element-wise with broadcasting.
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
/// * A `Tensor<FixedType>` instance representing the result of the element-wise addition with broadcasting.
fn FixedType_add_tensor(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
    check_compatibility(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_self = self.unravel_index(n);
        let indices_other = other.unravel_index(n);

        let i = broadcast_index_mapping(*self.shape, indices_self);
        let j = broadcast_index_mapping(*other.shape, indices_other);

        result.append(*(*self.data).at(i) + *(*other.data).at(j));

        n += 1;
        if n == (*self.data).len() {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span());
}

/// Subtracts two `Tensor<FixedType>` instances element-wise with broadcasting.
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
/// * A `Tensor<FixedType>` instance representing the result of the element-wise subtraction with broadcasting.
fn FixedType_sub_tensor(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
    check_compatibility(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_self = self.unravel_index(n);
        let indices_other = other.unravel_index(n);

        let i = broadcast_index_mapping(*self.shape, indices_self);
        let j = broadcast_index_mapping(*other.shape, indices_other);

        result.append(*(*self.data).at(i) - *(*other.data).at(j));

        n += 1;
        if n == (*self.data).len() {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span());
}

/// Multiplies two `Tensor<FixedType>` instances element-wise with broadcasting.
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
/// * A `Tensor<FixedType>` instance representing the result of the element-wise multiplication with broadcasting.
fn FixedType_mul_tensor(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
    check_compatibility(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_self = self.unravel_index(n);
        let indices_other = other.unravel_index(n);

        let i = broadcast_index_mapping(*self.shape, indices_self);
        let j = broadcast_index_mapping(*other.shape, indices_other);

        result.append(*(*self.data).at(i) * *(*other.data).at(j));

        n += 1;
        if n == (*self.data).len() {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span());
}

/// Divides two `Tensor<FixedType>` instances element-wise with broadcasting.
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
/// * A `Tensor<FixedType>` instance representing the result of the element-wise division with broadcasting.
fn FixedType_div_tensor(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
    check_compatibility(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_self = self.unravel_index(n);
        let indices_other = other.unravel_index(n);

        let i = broadcast_index_mapping(*self.shape, indices_self);
        let j = broadcast_index_mapping(*other.shape, indices_other);

        result.append(*(*self.data).at(i) / *(*other.data).at(j));

        n += 1;
        if n == (*self.data).len() {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span());
}

/// Reorders the axes of an FixedType tensor according to the given axes permutation.
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
/// * A `Tensor<FixedType>` instance with the axes reordered according to the given permutation.
fn FixedType_transpose(self: @Tensor<FixedType>, axes: Span<usize>) -> Tensor<FixedType> {
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

    return TensorTrait::<FixedType>::new(output_shape, output_data.span());
}
