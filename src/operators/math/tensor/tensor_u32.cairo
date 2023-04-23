use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

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
use onnx_cairo::utils::check_gas;

impl U32Tensor of TensorTrait<u32> {
    /// Creates tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - A reference to an array of usizes representing the shape of the tensor.
    /// * `data` -  A reference to an array of u32 reprensenting the data of the tensor as flat array.
    ///
    /// # Returns
    ///
    /// The tensor.
    ///
    /// # Panics
    ///
    /// Panic if the shape of the tensor does not match the size of the data array.
    fn new(shape: Span<usize>, data: @Array<u32>) -> Tensor<u32> {
        new_tensor(shape, data)
    }

    /// Returns the value of a particular element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    /// * `indices` - The indices of the element.
    ///
    /// # Returns
    ///
    /// The value of the element at the specified indices.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    fn at(self: @Tensor<u32>, indices: Span<usize>) -> u32 {
        u32_at_tensor(self, indices)
    }

    /// Returns the minimum value in the tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    ///
    /// # Returns
    ///
    /// The minimum value in tensor.
    // TODO: find minimum by axis
    fn min(self: @Tensor<u32>) -> u32 {
        u32_min_tensor(*self.data)
    }

    /// Returns the maximum value in the tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    ///
    /// # Returns
    ///
    /// The maximum value in tensor.
    // TODO: find maximum by axis
    fn max(self: @Tensor<u32>) -> u32 {
        u32_max_tensor(*self.data)
    }

    /// Returns the stride of a tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    ///
    /// # Returns
    ///
    /// the stride of a tensor.
    fn stride(self: @Tensor<u32>) -> Span<usize> {
        stride(*self.shape)
    }

    /// Returns the flat index corresponding to an array of indices.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    /// * `indices` - A reference to the indices.
    ///
    /// # Returns
    ///
    /// the flat index corresponding to an array of indices.
    fn ravel_index(self: @Tensor<u32>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    /// Returns the array of indices corresponding to a flat index.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    /// * `indices` - A reference to the indices.
    ///
    /// # Returns
    ///
    /// the array of indices corresponding to a flat index.
    fn unravel_index(self: @Tensor<u32>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    /// Gives a new shape to an array without changing its data.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    /// * `target_shape` - the new shape.
    ///
    /// # Returns
    ///
    /// the reshaped array.
    ///
    /// # Panics
    ///
    /// Panics if the target shape is not compatible to the original shape.
    fn reshape(self: @Tensor<u32>, target_shape: Span<usize>) -> Tensor<u32> {
        reshape(self, target_shape)
    }

    /// Computes the sum of elements across dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    /// * `axis` - The dimensions to reduce..
    ///
    /// # Returns
    ///
    /// The reduced tensor.
    ///
    /// # Panics
    ///
    /// Panic if the axis is larger than the dimension of the tensor.
    fn reduce_sum(self: @Tensor<u32>, axis: usize) -> Tensor<u32> {
        u32_reduce_sum(self, axis)
    }

    /// Computes the argmax of a tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    /// * `axis` - The dimension along which argmax is performed.
    ///
    /// # Returns
    ///
    /// Returns the indices of the maximum values along an axis.
    ///
    /// # Panics
    ///
    /// Panic if the axis is larger than the dimension of the tensor.
    fn argmax(self: @Tensor<u32>, axis: usize) -> Tensor<usize> {
        u32_argmax(self, axis)
    }

    /// Reorders the axes of an u32 tensor according to the given axes permutation.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    /// * `axes` - The axes permutation.
    ///
    /// # Returns
    ///
    /// Returns transposed tensor.
    fn transpose(self: @Tensor<u32>, axes: @Array<usize>) -> Tensor<u32> {
        u32_transpose(self, axes)
    }
}

impl U32TensorAdd of Add<Tensor<u32>> {
    fn add(self: Tensor<u32>, other: Tensor<u32>) -> Tensor<u32> {
        u32_add_tensor(@self, @other)
    }
}

impl U32TensorSub of Sub<Tensor<u32>> {
    fn sub(self: Tensor<u32>, other: Tensor<u32>) -> Tensor<u32> {
        u32_sub_tensor(@self, @other)
    }
}

impl U32TensorMul of Mul<Tensor<u32>> {
    fn mul(self: Tensor<u32>, other: Tensor<u32>) -> Tensor<u32> {
        u32_mul_tensor(@self, @other)
    }
}

impl U32TensorDiv of Div<Tensor<u32>> {
    fn div(self: Tensor<u32>, other: Tensor<u32>) -> Tensor<u32> {
        u32_div_tensor(@self, @other)
    }
}

fn u32_at_tensor(self: @Tensor<u32>, indices: Span<usize>) -> u32 {
    let data = *self.data;
    *data.at(self.ravel_index(indices))
}

fn u32_min_tensor(vec: @Array::<u32>) -> u32 {
    let mut min_value = 4294967295_u32;

    let mut i: usize = 0;
    loop {
        check_gas();

        if (min_value > *vec.at(i)) {
            min_value = *vec.at(i);
        }

        i += 1;
        if i == vec.len() {
            break ();
        };
    };

    return min_value;
}

fn u32_max_tensor(vec: @Array::<u32>) -> u32 {
    let mut max_value = 0_u32;
    let mut i: usize = 0;
    loop {
        check_gas();

        if (max_value < *vec.at(i)) {
            max_value = *vec.at(i);
        }

        i += 1;
        if i == vec.len() {
            break ();
        };
    };

    return max_value;
}

// --- BROADCAST OPERATIONS ---

fn u32_add_tensor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
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

    return TensorTrait::<u32>::new(*self.shape, @result);
}

fn u32_sub_tensor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
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

    return TensorTrait::<u32>::new(*self.shape, @result);
}

fn u32_mul_tensor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
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

    return TensorTrait::<u32>::new(*self.shape, @result);
}

fn u32_div_tensor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
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

    return TensorTrait::<u32>::new(*self.shape, @result);
}

// --- REDUCE OPERATIONS ---

// REDUCE SUM
// Sums the elements along the given axis of an u32 tensor
fn u32_reduce_sum(self: @Tensor<u32>, axis: usize) -> Tensor<u32> {
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

    return TensorTrait::<u32>::new(output_shape, @output_data);
}

// Helper function that accumulates the sum of elements along a specific axis
fn accumulate_sum(input: @Tensor<u32>, output_indices: Span<usize>, axis: usize) -> u32 {
    let axis_len = *(*input.shape).at(axis);
    let mut acc = 0_u32;

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

// ARGMAX
// Returns the indices of the maximum values along the given axis of an u32 tensor
fn u32_argmax(self: @Tensor<u32>, axis: usize) -> Tensor<usize> {
    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_output_shape(*self.shape, axis);
    let output_data_len = len_from_shape(output_shape);

    let mut index: usize = 0;
    loop {
        check_gas();

        let output_indices = unravel_index(index, output_shape);
        let current_argmax = find_argmax(self, output_indices, axis, 0, 0, 0);

        output_data.append(current_argmax);

        index += 1;
        if index == output_data_len {
            break ();
        };
    };

    return TensorTrait::<usize>::new(output_shape, @output_data);
}

// Recursive helper function that finds the index of the maximum value along a specific axis
fn find_argmax(
    input: @Tensor<u32>,
    output_indices: Span<usize>,
    axis: usize,
    axis_index: usize,
    max_value: u32,
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

// TRANSPOSE
// Reorders the axes of an u32 tensor according to the given axes permutation
fn u32_transpose(self: @Tensor<u32>, axes: @Array<usize>) -> Tensor<u32> {
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

    return TensorTrait::<u32>::new(output_shape, @output_data);
}
