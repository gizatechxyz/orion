use array::ArrayTrait;
use option::OptionTrait;

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

impl U32Tensor of TensorTrait::<u32> {
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
    fn new(shape: @Array<usize>, data: @Array<u32>) -> Tensor<u32> {
        u32_new_tensor(shape, data)
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
    fn at(self: @Tensor<u32>, indices: @Array<usize>) -> u32 {
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
    fn stride(self: @Tensor<u32>) -> Array<usize> {
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
    fn ravel_index(self: @Tensor<u32>, indices: @Array<usize>) -> usize {
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
    fn unravel_index(self: @Tensor<u32>, index: usize) -> Array<usize> {
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
        i32_argmax(self, axis)
    }
}

impl U32TensorAdd of Add::<Tensor<u32>> {
    fn add(self: Tensor<u32>, other: Tensor<u32>) -> Tensor<u32> {
        u32_add_tensor(@self, @other)
    }
}

impl U32TensorSub of Sub::<Tensor<u32>> {
    fn sub(self: Tensor<u32>, other: Tensor<u32>) -> Tensor<u32> {
        u32_sub_tensor(@self, @other)
    }
}

impl U32TensorMul of Mul::<Tensor<u32>> {
    fn mul(self: Tensor<u32>, other: Tensor<u32>) -> Tensor<u32> {
        u32_mul_tensor(@self, @other)
    }
}

impl U32TensorDiv of Div::<Tensor<u32>> {
    fn div(self: Tensor<u32>, other: Tensor<u32>) -> Tensor<u32> {
        u32_div_tensor(@self, @other)
    }
}

fn u32_new_tensor(shape: @Array<usize>, data: @Array<u32>) -> Tensor<u32> {
    check_shape::<u32>(shape, data);
    Tensor::<u32> { shape, data }
}

#[inline(always)]
fn u32_at_tensor(self: @Tensor<u32>, indices: @Array<usize>) -> u32 {
    let data = *self.data;
    *data.at(self.ravel_index(indices))
}

fn u32_min_tensor(vec: @Array::<u32>) -> u32 {
    let mut min_value = 4294967295_u32;
    __u32_min_tensor(vec, ref min_value, 0_usize);
    return min_value;
}

fn __u32_min_tensor(vec: @Array::<u32>, ref min_value: u32, n: usize) {
    check_gas();
    if n == vec.len() {
        return ();
    }

    if (min_value > *vec.at(n)) {
        min_value = *vec.at(n);
    }

    __u32_min_tensor(vec, ref min_value, n + 1_usize);
}

fn u32_max_tensor(vec: @Array::<u32>) -> u32 {
    let mut max_value = 0_u32;
    __u32_max_tensor(vec, ref max_value, 0_usize);
    return max_value;
}

fn __u32_max_tensor(vec: @Array::<u32>, ref max_value: u32, n: usize) {
    check_gas();
    if n == vec.len() {
        return ();
    }

    if (max_value < *vec.at(n)) {
        max_value = *vec.at(n);
    }

    __u32_max_tensor(vec, ref max_value, n + 1_usize);
}

// --- BROADCAST OPERATIONS ---

fn u32_add_tensor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
    check_compatibility(*self.shape, *other.shape, 0_usize);
    let mut result = ArrayTrait::new();
    __u32_add_tensor(self, other, ref result, 0_usize);
    return TensorTrait::<u32>::new(*self.shape, @result);
}

fn __u32_add_tensor(self: @Tensor<u32>, other: @Tensor<u32>, ref result: Array::<u32>, n: usize) {
    check_gas();
    if n == (*self.data).len() {
        return ();
    }

    let indices_self = self.unravel_index(n);
    let indices_other = other.unravel_index(n);

    let i = broadcast_index_mapping(*self.shape, @indices_self);
    let j = broadcast_index_mapping(*other.shape, @indices_other);

    result.append(*(*self.data).at(i) + *(*other.data).at(j));
    __u32_add_tensor(self, other, ref result, n + 1_usize);
}

fn u32_sub_tensor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
    check_compatibility(*self.shape, *other.shape, 0_usize);
    let mut result = ArrayTrait::new();
    __u32_sub_tensor(self, other, ref result, 0_usize);
    return TensorTrait::<u32>::new(*self.shape, @result);
}

fn __u32_sub_tensor(self: @Tensor<u32>, other: @Tensor<u32>, ref result: Array::<u32>, n: usize) {
    check_gas();
    if n == (*self.data).len() {
        return ();
    }

    let indices_self = self.unravel_index(n);
    let indices_other = other.unravel_index(n);

    let i = broadcast_index_mapping(*self.shape, @indices_self);
    let j = broadcast_index_mapping(*other.shape, @indices_other);

    result.append(*(*self.data).at(i) - *(*other.data).at(j));
    __u32_sub_tensor(self, other, ref result, n + 1_usize);
}

fn u32_mul_tensor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
    check_compatibility(*self.shape, *other.shape, 0_usize);
    let mut result = ArrayTrait::new();
    __u32_mul_tensor(self, other, ref result, 0_usize);
    return TensorTrait::<u32>::new(*self.shape, @result);
}

fn __u32_mul_tensor(self: @Tensor<u32>, other: @Tensor<u32>, ref result: Array::<u32>, n: usize) {
    check_gas();
    if n == (*self.data).len() {
        return ();
    }

    let indices_self = self.unravel_index(n);
    let indices_other = other.unravel_index(n);

    let i = broadcast_index_mapping(*self.shape, @indices_self);
    let j = broadcast_index_mapping(*other.shape, @indices_other);

    result.append(*(*self.data).at(i) * *(*other.data).at(j));
    __u32_mul_tensor(self, other, ref result, n + 1_usize);
}

fn u32_div_tensor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
    check_compatibility(*self.shape, *other.shape, 0_usize);
    let mut result = ArrayTrait::new();
    __u32_div_tensor(self, other, ref result, 0_usize);
    return TensorTrait::<u32>::new(*self.shape, @result);
}

fn __u32_div_tensor(self: @Tensor<u32>, other: @Tensor<u32>, ref result: Array::<u32>, n: usize) {
    check_gas();
    if n == (*self.data).len() {
        return ();
    }

    let indices_self = self.unravel_index(n);
    let indices_other = other.unravel_index(n);

    let i = broadcast_index_mapping(*self.shape, @indices_self);
    let j = broadcast_index_mapping(*other.shape, @indices_other);

    result.append(*(*self.data).at(i) / *(*other.data).at(j));
    __u32_div_tensor(self, other, ref result, n + 1_usize);
}

// --- REDUCE OPERATIONS ---

fn u32_reduce_sum(self: @Tensor<u32>, axis: usize) -> Tensor<u32> {
    let mut output_shape = ArrayTrait::new();
    let mut output_data = ArrayTrait::new();

    reduce_helper(*self.shape, axis, ref output_shape, 0_usize);
    __u32_reduce_sum(
        self, @output_shape, len_from_shape(@output_shape, 0_usize), axis, ref output_data, 0_usize
    );

    return TensorTrait::<u32>::new(@output_shape, @output_data);
}

fn __u32_reduce_sum(
    self: @Tensor<u32>,
    output_shape: @Array<usize>,
    output_data_len: usize,
    axis: usize,
    ref output_data: Array<u32>,
    n: usize
) {
    check_gas();

    if n == output_data_len {
        return ();
    }

    let output_indices = unravel_index(n, output_shape);
    let current_sum = accumulate_sum_recursive(self, @output_indices, axis, 0_usize);

    output_data.append(current_sum);
    __u32_reduce_sum(self, output_shape, output_data_len, axis, ref output_data, n + 1_usize);
}

fn accumulate_sum_recursive(
    input: @Tensor<u32>, output_indices: @Array<usize>, axis: usize, axis_index: usize, 
) -> u32 {
    check_gas();

    if axis_index == *(*input.shape).at(axis) {
        return 0_u32;
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

fn i32_argmax(self: @Tensor<u32>, axis: usize) -> Tensor<usize> {
    let mut output_shape = ArrayTrait::new();
    let mut output_data = ArrayTrait::new();

    reduce_helper(*self.shape, axis, ref output_shape, 0_usize);
    __i32_argmax(
        self, @output_shape, len_from_shape(@output_shape, 0_usize), axis, ref output_data, 0_usize
    );

    return TensorTrait::<usize>::new(@output_shape, @output_data);
}

fn __i32_argmax(
    self: @Tensor<u32>,
    output_shape: @Array<usize>,
    output_data_len: usize,
    axis: usize,
    ref output_data: Array<usize>,
    n: usize
) {
    check_gas();

    if n == output_data_len {
        return ();
    }

    let output_indices = unravel_index(n, output_shape);
    let current_argmax = accumulate_argmax_recursive(
        self, @output_indices, axis, 0_usize, 0_u32, 0_usize
    );

    output_data.append(current_argmax);
    __i32_argmax(self, output_shape, output_data_len, axis, ref output_data, n + 1_usize);
}

fn accumulate_argmax_recursive(
    input: @Tensor<u32>,
    output_indices: @Array<usize>,
    axis: usize,
    axis_index: usize,
    max_value: u32,
    max_index: usize
) -> usize {
    check_gas();

    if axis_index == *(*input.shape).at(axis) {
        return max_index;
    }

    let mut input_indices = ArrayTrait::new();
    combine_indices(output_indices, axis_index, axis, ref input_indices, 0_usize);
    let input_index = ravel_index(*input.shape, @input_indices);
    let ele = *(*input.data).at(input_index);

    if ele > max_value {
        let max_value = ele;
        let max_index = axis_index;
        return accumulate_argmax_recursive(
            input, output_indices, axis, axis_index + 1_usize, max_value, max_index
        );
    }

    return accumulate_argmax_recursive(
        input, output_indices, axis, axis_index + 1_usize, max_value, max_index
    );
}
