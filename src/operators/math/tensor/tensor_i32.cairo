//! This module defines and implement a Tensor for i32 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;
use onnx_cairo::operators::math::tensor::helpers::check_shape;
use onnx_cairo::operators::math::tensor::helpers::check_compatibility;
use onnx_cairo::operators::math::tensor::core::new_tensor;
use onnx_cairo::operators::math::tensor::core::stride;
use onnx_cairo::operators::math::tensor::core::Tensor;
use onnx_cairo::operators::math::tensor::core::TensorTrait;
use onnx_cairo::operators::math::tensor::core::ravel_index;
use onnx_cairo::operators::math::tensor::core::unravel_index;
use onnx_cairo::operators::math::tensor::core::reshape;
use onnx_cairo::operators::math::tensor::helpers::broadcast_index_mapping;
use onnx_cairo::operators::math::tensor::helpers::reduce_output_shape;
use onnx_cairo::operators::math::tensor::helpers::len_from_shape;
use onnx_cairo::operators::math::tensor::helpers::combine_indices;
use onnx_cairo::operators::math::tensor::helpers::find_axis;
use onnx_cairo::operators::math::tensor::helpers::permutation_output_shape;
use onnx_cairo::operators::math::tensor::tensor_u32;
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
        i32_min_tensor(*self.data)
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
        i32_max_tensor(*self.data)
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
        i32_reduce_sum(self, axis)
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
        i32_argmax(self, axis)
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

/// Finds the minimum value in a `Tensor<i32>` array.
///
/// # Arguments
/// * `vec` -  A span containing the data array of i32 elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An i32 value representing the minimum value in the array.
fn i32_min_tensor(vec: Span::<i32>) -> i32 {
    let mut min_value: i32 = IntegerTrait::new(2147483647_u32, false);

    let mut i: usize = 0;
    loop {
        check_gas();

        let check_min = min_value.min(*vec.at(i));
        if (min_value > check_min) {
            min_value = check_min;
        }

        i += 1;
        if i == vec.len() {
            break ();
        };
    };

    return min_value;
}

/// Finds the maximum value in a `Tensor<i32>` array.
///
/// # Arguments
/// * `vec` -  A span containing the data array of i32 elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An i32 value representing the maximum value in the array.
fn i32_max_tensor(vec: Span::<i32>) -> i32 {
    let mut max_value: i32 = IntegerTrait::new(0_u32, false);

    let mut i: usize = 0;
    loop {
        check_gas();

        let check_max = max_value.max(*vec.at(i));
        if (max_value < check_max) {
            max_value = check_max;
        }

        i += 1;
        if i == vec.len() {
            break ();
        };
    };

    return max_value;
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

    return TensorTrait::<i32>::new(*self.shape, result.span());
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

    return TensorTrait::<i32>::new(*self.shape, result.span());
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

    return TensorTrait::<i32>::new(*self.shape, result.span());
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

    return TensorTrait::<i32>::new(*self.shape, result.span());
}

/// --- REDUCE OPERATIONS ---

/// Sums the elements along the given axis of an i32 tensor.
///
/// # Arguments
/// * `self` - The input tensor.
/// * `axis` - The axis along which to sum the elements.
///
/// # Panics
/// * Panics if axis is not in the range of the input tensor's dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<i32>` instance representing the result of the reduction.
fn i32_reduce_sum(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
    assert(axis <= (*self.shape).len(), 'axis out of dimensions');
    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_output_shape(*self.shape, axis);
    let output_data_len = len_from_shape(output_shape);

    let mut index: usize = 0;
    loop {
        check_gas();

        let output_indices = unravel_index(index, output_shape);
        let current_sum = accumulate_sum(self, output_indices, axis);

        output_data.append(current_sum);

        index += 1;
        if index == output_data_len {
            break ();
        };
    };

    return TensorTrait::<i32>::new(output_shape, output_data.span());
}

/// Helper function that accumulates the sum of elements along a specific axis.
///
/// # Arguments
/// * `input` - The input tensor.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to accumulate the sum.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An i32 value representing the accumulated sum along the specified axis.
fn accumulate_sum(input: @Tensor<i32>, output_indices: Span<usize>, axis: usize) -> i32 {
    let axis_len = *(*input.shape).at(axis);
    let mut acc = IntegerTrait::new(0_u32, false);

    let mut axis_index: usize = 0;
    loop {
        check_gas();

        if axis_index == axis_len {
            break ();
        }

        let input_indices = combine_indices(output_indices, axis_index, axis);
        let input_index = ravel_index(*input.shape, input_indices);
        let ele = *(*input.data).at(input_index);

        acc += ele;
        axis_index += 1;
    };

    return acc;
}

/// Returns the indices of the maximum values along the given axis of an i32 tensor.
///
/// # Arguments
/// * `self` - The input tensor.
/// * `axis` - The axis along which to find the maximum values.
///
/// # Panics
/// * Panics if axis is not in the range of the input tensor's dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<usize>` instance representing the indices of the maximum values along the given axis.
fn i32_argmax(self: @Tensor<i32>, axis: usize) -> Tensor<usize> {
    assert(axis <= (*self.shape).len(), 'axis out of dimensions');

    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_output_shape(*self.shape, axis);
    let output_data_len = len_from_shape(output_shape);

    let mut index: usize = 0;
    loop {
        check_gas();

        let output_indices = unravel_index(index, output_shape);
        let current_argmax = find_argmax(
            self, output_indices, axis, 0, IntegerTrait::new(2147483648, true), 0
        );

        output_data.append(current_argmax);

        index += 1;
        if index == output_data_len {
            break ();
        };
    };

    return TensorTrait::<usize>::new(output_shape, output_data.span());
}

/// Recursive helper function that finds the index of the maximum value along a specific axis.
///
/// # Arguments
/// * `input` - The input tensor.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to find the maximum value.
/// * `axis_index` - The current index along the specified axis.
/// * `max_value` - The current maximum value found along the axis.
/// * `argmax` - The current index of the maximum value along the axis.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize value representing the index of the maximum value along the specified axis.
fn find_argmax(
    input: @Tensor<i32>,
    output_indices: Span<usize>,
    axis: usize,
    axis_index: usize,
    max_value: i32,
    argmax: usize
) -> usize {
    check_gas();

    if axis_index == *(*input.shape).at(axis) {
        return argmax;
    }

    let input_indices = combine_indices(output_indices, axis_index, axis);
    let input_index = ravel_index(*input.shape, input_indices);
    let ele = *(*input.data).at(input_index);

    let (new_max_value, new_argmax) = if ele > max_value {
        (ele, axis_index)
    } else {
        (max_value, argmax)
    };

    return find_argmax(
        input, output_indices, axis, axis_index + 1_usize, new_max_value, new_argmax
    );
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
