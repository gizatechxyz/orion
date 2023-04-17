use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;
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
use onnx_cairo::operators::math::tensor::tensor_u32;

use onnx_cairo::utils::check_gas;

impl i32Tensor of TensorTrait<i32> {
    /// Creates tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - A reference to an array of usizes representing the shape of the tensor.
    /// * `data` -  A reference to an array of i32 reprensenting the data of the tensor as flat array.
    ///
    /// # Returns
    ///
    /// The tensor.
    ///
    /// # Panics
    ///
    /// Panic if the shape of the tensor does not match the size of the data array.
    fn new(shape: @Array<usize>, data: @Array<i32>) -> Tensor<i32> {
        i32_new_tensor(shape, data)
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
    fn at(self: @Tensor<i32>, indices: @Array<usize>) -> i32 {
        i32_at_tensor(self, indices)
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
    fn min(self: @Tensor<i32>) -> i32 {
        i32_min_tensor(*self.data)
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
    fn max(self: @Tensor<i32>) -> i32 {
        i32_max_tensor(*self.data)
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
    fn stride(self: @Tensor<i32>) -> Array<usize> {
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
    fn ravel_index(self: @Tensor<i32>, indices: @Array<usize>) -> usize {
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
    fn unravel_index(self: @Tensor<i32>, index: usize) -> Array<usize> {
        unravel_index(index, *self.shape)
    }

    /// Computes the sum of elements across dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the tensor.
    /// * `axis` - The dimension to reduce..
    ///
    /// # Returns
    ///
    /// The reduced tensor.
    ///
    /// # Panics
    ///
    /// Panic if the axis is larger than the dimension of the tensor.
    fn reduce_sum(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
        i32_reduce_sum(self, axis)
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
    fn argmax(self: @Tensor<i32>, axis: usize) -> Tensor<usize> {
        i32_argmax(self, axis)
    }
}

impl i32TensorAdd of Add<Tensor<i32>> {
    fn add(self: Tensor<i32>, other: Tensor<i32>) -> Tensor<i32> {
        i32_add_tensor(@self, @other)
    }
}

impl i32TensorSub of Sub<Tensor<i32>> {
    fn sub(self: Tensor<i32>, other: Tensor<i32>) -> Tensor<i32> {
        i32_sub_tensor(@self, @other)
    }
}

impl i32TensorMul of Mul<Tensor<i32>> {
    fn mul(self: Tensor<i32>, other: Tensor<i32>) -> Tensor<i32> {
        i32_mul_tensor(@self, @other)
    }
}

impl i32TensorDiv of Div<Tensor<i32>> {
    fn div(self: Tensor<i32>, other: Tensor<i32>) -> Tensor<i32> {
        i32_div_tensor(@self, @other)
    }
}

fn i32_new_tensor(shape: @Array<usize>, data: @Array<i32>) -> Tensor<i32> {
    check_shape::<i32>(shape, data);
    Tensor::<i32> { shape, data }
}

#[inline(always)]
fn i32_at_tensor(self: @Tensor<i32>, indices: @Array<usize>) -> i32 {
    let data = *self.data;
    *data.at(self.ravel_index(indices))
}

fn i32_min_tensor(vec: @Array::<i32>) -> i32 {
    let mut min_value: i32 = IntegerTrait::new(65535_u32, false);

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


fn i32_max_tensor(vec: @Array::<i32>) -> i32 {
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

fn i32_add_tensor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
    check_compatibility(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_self = self.unravel_index(n);
        let indices_other = other.unravel_index(n);

        let i = broadcast_index_mapping(*self.shape, @indices_self);
        let j = broadcast_index_mapping(*other.shape, @indices_other);

        result.append(*(*self.data).at(i) + *(*other.data).at(j));

        n += 1;
        if n == (*self.data).len() {
            break ();
        };
    };

    return TensorTrait::<i32>::new(*self.shape, @result);
}

fn i32_sub_tensor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
    check_compatibility(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_self = self.unravel_index(n);
        let indices_other = other.unravel_index(n);

        let i = broadcast_index_mapping(*self.shape, @indices_self);
        let j = broadcast_index_mapping(*other.shape, @indices_other);

        result.append(*(*self.data).at(i) - *(*other.data).at(j));

        n += 1;
        if n == (*self.data).len() {
            break ();
        };
    };

    return TensorTrait::<i32>::new(*self.shape, @result);
}

fn i32_mul_tensor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
    check_compatibility(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_self = self.unravel_index(n);
        let indices_other = other.unravel_index(n);

        let i = broadcast_index_mapping(*self.shape, @indices_self);
        let j = broadcast_index_mapping(*other.shape, @indices_other);

        result.append(*(*self.data).at(i) * *(*other.data).at(j));

        n += 1;
        if n == (*self.data).len() {
            break ();
        };
    };

    return TensorTrait::<i32>::new(*self.shape, @result);
}

fn i32_div_tensor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
    check_compatibility(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_self = self.unravel_index(n);
        let indices_other = other.unravel_index(n);

        let i = broadcast_index_mapping(*self.shape, @indices_self);
        let j = broadcast_index_mapping(*other.shape, @indices_other);

        result.append(*(*self.data).at(i) / *(*other.data).at(j));

        n += 1;
        if n == (*self.data).len() {
            break ();
        };
    };

    return TensorTrait::<i32>::new(*self.shape, @result);
}

// --- REDUCE OPERATIONS ---

// REDUCE SUM
fn i32_reduce_sum(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_helper(*self.shape, axis);
    __i32_reduce_sum(
        self, @output_shape, len_from_shape(@output_shape), axis, ref output_data, 0_usize
    );

    return TensorTrait::<i32>::new(@output_shape, @output_data);
}

fn __i32_reduce_sum(
    self: @Tensor<i32>,
    output_shape: @Array<usize>,
    output_data_len: usize,
    axis: usize,
    ref output_data: Array<i32>,
    n: usize
) {
    check_gas();

    if n == output_data_len {
        return ();
    }

    let output_indices = unravel_index(n, output_shape);
    let current_sum = accumulate_sum_recursive(self, @output_indices, axis, 0_usize);

    output_data.append(current_sum);
    __i32_reduce_sum(self, output_shape, output_data_len, axis, ref output_data, n + 1_usize);
}

fn accumulate_sum_recursive(
    input: @Tensor<i32>, output_indices: @Array<usize>, axis: usize, axis_index: usize, 
) -> i32 {
    check_gas();

    if axis_index == *(*input.shape).at(axis) {
        return IntegerTrait::new(0_u32, false);
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

// ARGMAX
fn i32_argmax(self: @Tensor<i32>, axis: usize) -> Tensor<usize> {
    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_helper(*self.shape, axis);
    __i32_argmax(
        self, @output_shape, len_from_shape(@output_shape), axis, ref output_data, 0_usize
    );

    return TensorTrait::<usize>::new(@output_shape, @output_data);
}

fn __i32_argmax(
    self: @Tensor<i32>,
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
    let current_argmax = accumulate_argmax(
        self, @output_indices, axis, 0_usize, IntegerTrait::new(2147483647_usize, false), 0_usize
    );

    output_data.append(current_argmax);
    __i32_argmax(self, output_shape, output_data_len, axis, ref output_data, n + 1_usize);
}

fn accumulate_argmax(
    input: @Tensor<i32>,
    output_indices: @Array<usize>,
    axis: usize,
    axis_index: usize,
    max_value: i32,
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
        return accumulate_argmax(
            input, output_indices, axis, axis_index + 1_usize, max_value, max_index
        );
    }

    return accumulate_argmax(
        input, output_indices, axis, axis_index + 1_usize, max_value, max_index
    );
}
