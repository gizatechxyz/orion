use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;
use onnx_cairo::operators::math::tensor::helpers::check_shape;
use onnx_cairo::operators::math::tensor::helpers::check_compatibility;
use onnx_cairo::operators::math::tensor::core::stride;
use onnx_cairo::operators::math::tensor::core::Tensor;
use onnx_cairo::operators::math::tensor::core::TensorTrait;
use onnx_cairo::operators::math::tensor::core::ravel_index;
use onnx_cairo::operators::math::tensor::core::unravel_index;
use onnx_cairo::operators::math::tensor::helpers::broadcast_index_mapping;
use onnx_cairo::operators::math::tensor::helpers::reduce_helper;
use onnx_cairo::operators::math::tensor::helpers::len_from_shape;
use onnx_cairo::utils::check_gas;

impl I33Tensor of TensorTrait::<i33> {
    /// Creates tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - A reference to an array of usizes representing the shape of the tensor.
    /// * `data` -  A reference to an array of i33 reprensenting the data of the tensor as flat array.
    ///
    /// # Returns
    ///
    /// The tensor.
    ///
    /// # Panics
    ///
    /// Panic if the shape of the tensor does not match the size of the data array.
    fn new(shape: @Array<usize>, data: @Array<i33>) -> Tensor<i33> {
        i33_new_tensor(shape, data)
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
    fn at(self: @Tensor<i33>, indices: @Array<usize>) -> i33 {
        i33_at_tensor(self, indices)
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
    fn min(self: @Tensor<i33>) -> i33 {
        i33_min_tensor(*self.data)
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
    fn max(self: @Tensor<i33>) -> i33 {
        i33_max_tensor(*self.data)
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
    fn stride(self: @Tensor<i33>) -> Array<usize> {
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
    fn ravel_index(self: @Tensor<i33>, indices: @Array<usize>) -> usize {
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
    fn unravel_index(self: @Tensor<i33>, index: usize) -> Array<usize> {
        unravel_index(index, *self.shape)
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
    fn reduce_sum(self: @Tensor<i33>, axis: usize) -> Tensor<i33> {
        i33_reduce_sum(self, axis)
    }
}

impl i33TensorAdd of Add::<Tensor<i33>> {
    fn add(self: Tensor<i33>, other: Tensor<i33>) -> Tensor<i33> {
        i33_add_tensor(@self, @other)
    }
}

impl i33TensorSub of Sub::<Tensor<i33>> {
    fn sub(self: Tensor<i33>, other: Tensor<i33>) -> Tensor<i33> {
        i33_sub_tensor(@self, @other)
    }
}

impl i33TensorMul of Mul::<Tensor<i33>> {
    fn mul(self: Tensor<i33>, other: Tensor<i33>) -> Tensor<i33> {
        i33_mul_tensor(@self, @other)
    }
}

impl i33TensorDiv of Div::<Tensor<i33>> {
    fn div(self: Tensor<i33>, other: Tensor<i33>) -> Tensor<i33> {
        i33_div_tensor(@self, @other)
    }
}

fn i33_new_tensor(shape: @Array<usize>, data: @Array<i33>) -> Tensor<i33> {
    check_shape::<i33>(shape, data);
    Tensor::<i33> { shape, data }
}

#[inline(always)]
fn i33_at_tensor(self: @Tensor<i33>, indices: @Array<usize>) -> i33 {
    let data = *self.data;
    *data.at(self.ravel_index(indices))
}

fn i33_min_tensor(vec: @Array::<i33>) -> i33 {
    let mut min_value = i33 { inner: 65535_u32, sign: false };
    __i33_min_tensor(vec, ref min_value, 0_usize);
    return min_value;
}

fn __i33_min_tensor(vec: @Array::<i33>, ref min_value: i33, n: usize) {
    check_gas();
    if n == vec.len() {
        return ();
    }

    let check_min = int33::min(min_value, *vec.at(n));
    if (min_value > check_min) {
        min_value = check_min;
    }

    __i33_min_tensor(vec, ref min_value, n + 1_usize);
}

fn i33_max_tensor(vec: @Array::<i33>) -> i33 {
    let mut max_value = i33 { inner: 0_u32, sign: false };
    __i33_max_tensor(vec, ref max_value, 0_usize);
    return max_value;
}

fn __i33_max_tensor(vec: @Array::<i33>, ref max_value: i33, n: usize) {
    check_gas();
    if n == vec.len() {
        return ();
    }

    let check_max = int33::max(max_value, *vec.at(n));
    if (max_value < check_max) {
        max_value = check_max;
    }

    __i33_max_tensor(vec, ref max_value, n + 1_usize);
}

// --- BROADCAST OPERATIONS ---

fn i33_add_tensor(self: @Tensor<i33>, other: @Tensor<i33>) -> Tensor<i33> {
    check_compatibility(*self.shape, *other.shape, 0_usize);
    let mut result = ArrayTrait::new();
    __i33_add_tensor(self, other, ref result, 0_usize);
    return TensorTrait::<i33>::new(*self.shape, @result);
}

fn __i33_add_tensor(self: @Tensor<i33>, other: @Tensor<i33>, ref result: Array::<i33>, n: usize) {
    check_gas();
    if n == (*self.data).len() {
        return ();
    }

    let indices_self = self.unravel_index(n);
    let indices_other = other.unravel_index(n);

    let i = broadcast_index_mapping(*self.shape, @indices_self);
    let j = broadcast_index_mapping(*other.shape, @indices_other);

    result.append(*(*self.data).at(i) + *(*other.data).at(j));
    __i33_add_tensor(self, other, ref result, n + 1_usize);
}

fn i33_sub_tensor(self: @Tensor<i33>, other: @Tensor<i33>) -> Tensor<i33> {
    check_compatibility(*self.shape, *other.shape, 0_usize);
    let mut result = ArrayTrait::new();
    __i33_sub_tensor(self, other, ref result, 0_usize);
    return TensorTrait::<i33>::new(*self.shape, @result);
}

fn __i33_sub_tensor(self: @Tensor<i33>, other: @Tensor<i33>, ref result: Array::<i33>, n: usize) {
    check_gas();
    if n == (*self.data).len() {
        return ();
    }

    let indices_self = self.unravel_index(n);
    let indices_other = other.unravel_index(n);

    let i = broadcast_index_mapping(*self.shape, @indices_self);
    let j = broadcast_index_mapping(*other.shape, @indices_other);

    result.append(*(*self.data).at(i) - *(*other.data).at(j));
    __i33_sub_tensor(self, other, ref result, n + 1_usize);
}

fn i33_mul_tensor(self: @Tensor<i33>, other: @Tensor<i33>) -> Tensor<i33> {
    check_compatibility(*self.shape, *other.shape, 0_usize);
    let mut result = ArrayTrait::new();
    __i33_mul_tensor(self, other, ref result, 0_usize);
    return TensorTrait::<i33>::new(*self.shape, @result);
}

fn __i33_mul_tensor(self: @Tensor<i33>, other: @Tensor<i33>, ref result: Array::<i33>, n: usize) {
    check_gas();
    if n == (*self.data).len() {
        return ();
    }

    let indices_self = self.unravel_index(n);
    let indices_other = other.unravel_index(n);

    let i = broadcast_index_mapping(*self.shape, @indices_self);
    let j = broadcast_index_mapping(*other.shape, @indices_other);

    result.append(*(*self.data).at(i) * *(*other.data).at(j));
    __i33_mul_tensor(self, other, ref result, n + 1_usize);
}

fn i33_div_tensor(self: @Tensor<i33>, other: @Tensor<i33>) -> Tensor<i33> {
    check_compatibility(*self.shape, *other.shape, 0_usize);
    let mut result = ArrayTrait::new();
    __i33_div_tensor(self, other, ref result, 0_usize);
    return TensorTrait::<i33>::new(*self.shape, @result);
}

fn __i33_div_tensor(self: @Tensor<i33>, other: @Tensor<i33>, ref result: Array::<i33>, n: usize) {
    check_gas();
    if n == (*self.data).len() {
        return ();
    }

    let indices_self = self.unravel_index(n);
    let indices_other = other.unravel_index(n);

    let i = broadcast_index_mapping(*self.shape, @indices_self);
    let j = broadcast_index_mapping(*other.shape, @indices_other);

    result.append(*(*self.data).at(i) / *(*other.data).at(j));
    __i33_div_tensor(self, other, ref result, n + 1_usize);
}

// --- REDUCE SUM ---

fn i33_reduce_sum(self: @Tensor<i33>, axis: usize) -> Tensor<i33> {
    let mut output_shape = ArrayTrait::new();
    let mut output_data = ArrayTrait::new();

    reduce_helper(*self.shape, axis, ref output_shape, 0_usize);
    __i33_reduce_sum(
        self, @output_shape, len_from_shape(@output_shape, 0_usize), axis, ref output_data, 0_usize
    );

    return TensorTrait::<i33>::new(@output_shape, @output_data);
}

fn __i33_reduce_sum(
    self: @Tensor<i33>,
    output_shape: @Array<usize>,
    output_data_len: usize,
    axis: usize,
    ref output_data: Array<i33>,
    n: usize
) {
    check_gas();

    if n == output_data_len {
        return ();
    }

    let output_indices = unravel_index(n, output_shape);
    let current_sum = accumulate_sum_recursive(self, @output_indices, axis, 0_usize);

    output_data.append(current_sum);
    __i33_reduce_sum(self, output_shape, output_data_len, axis, ref output_data, n + 1_usize);
}

fn accumulate_sum_recursive(
    input: @Tensor<i33>, output_indices: @Array<usize>, axis: usize, axis_index: usize, 
) -> i33 {
    check_gas();

    if axis_index == *(*input.shape).at(axis) {
        return i33 { inner: 0_usize, sign: false };
    }

    let mut input_indices = ArrayTrait::new();
    combine_indices(output_indices, axis_index, axis, ref input_indices, 0_usize);
    let input_index = ravel_index(*input.shape, @input_indices);
    let ele = *(*input.data).at(input_index);

    let acc = accumulate_sum_recursive(input, output_indices, axis, axis_index + 1_usize);

    return ele + acc;
}

// TODO to be removed when managed by slicing
fn combine_indices(
    output_indices: @Array<usize>,
    axis_index: usize,
    axis: usize,
    ref result: Array<usize>,
    n: usize
) {
    check_gas();

    if n > output_indices.len() {
        return ();
    }

    if n == axis {
        result.append(axis_index);
    } else if n > axis {
        result.append(*output_indices.at(n - 1_usize));
    } else {
        result.append(*output_indices.at(n));
    }

    combine_indices(output_indices, axis_index, axis, ref result, n + 1_usize);
}
