use array::ArrayTrait;
use array::SpanTrait;
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
use onnx_cairo::operators::math::tensor::helpers::combine_indices;
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
    fn new(shape: Span<usize>, data: @Array<i32>) -> Tensor<i32> {
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
    fn at(self: @Tensor<i32>, indices: Span<usize>) -> i32 {
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
    fn stride(self: @Tensor<i32>) -> Span<usize> {
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
    fn ravel_index(self: @Tensor<i32>, indices: Span<usize>) -> usize {
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
    fn unravel_index(self: @Tensor<i32>, index: usize) -> Span<usize> {
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

fn i32_new_tensor(shape: Span<usize>, data: @Array<i32>) -> Tensor<i32> {
    check_shape::<i32>(shape, data);
    Tensor::<i32> { shape, data }
}

fn i32_at_tensor(self: @Tensor<i32>, indices: Span<usize>) -> i32 {
    let data = *self.data;
    *data.at(self.ravel_index(indices))
}

fn i32_min_tensor(vec: @Array::<i32>) -> i32 {
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

        let i = broadcast_index_mapping(*self.shape, indices_self);
        let j = broadcast_index_mapping(*other.shape, indices_other);

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

        let i = broadcast_index_mapping(*self.shape, indices_self);
        let j = broadcast_index_mapping(*other.shape, indices_other);

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

        let i = broadcast_index_mapping(*self.shape, indices_self);
        let j = broadcast_index_mapping(*other.shape, indices_other);

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

        let i = broadcast_index_mapping(*self.shape, indices_self);
        let j = broadcast_index_mapping(*other.shape, indices_other);

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

    return TensorTrait::<i32>::new(output_shape, @output_data);
}

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

// ARGMAX
fn i32_argmax(self: @Tensor<i32>, axis: usize) -> Tensor<usize> {
    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_helper(*self.shape, axis);
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

    return TensorTrait::<usize>::new(output_shape, @output_data);
}

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

