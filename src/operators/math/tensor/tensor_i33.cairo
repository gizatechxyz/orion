use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;
use onnx_cairo::operators::math::tensor::helpers::check_shape;
use onnx_cairo::operators::math::tensor::helpers::check_compatibility;
use onnx_cairo::operators::math::tensor::core::stride;
use onnx_cairo::operators::math::tensor::core::Tensor;
use onnx_cairo::operators::math::tensor::core::TensorTrait;
use onnx_cairo::operators::math::tensor::core::len_from_shape;
use onnx_cairo::operators::math::tensor::core::ravel_index;
use onnx_cairo::operators::math::tensor::core::unravel_index;
use onnx_cairo::operators::math::tensor::core::broadcast_index_mapping;
use onnx_cairo::utils::check_gas;

impl I33Tensor of TensorTrait::<i33> {
    fn new(shape: @Array<usize>, data: @Array<i33>) -> Tensor<i33> {
        i33_new_tensor(shape, data)
    }

    fn at(self: @Tensor<i33>, indices: @Array<usize>) -> i33 {
        i33_at_tensor(self, indices)
    }

    fn min(self: @Tensor<i33>) -> i33 {
        i33_min_tensor(*self.data)
    }

    fn max(self: @Tensor<i33>) -> i33 {
        i33_max_tensor(*self.data)
    }

    fn stride(self: @Tensor<i33>) -> Array<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<i33>, indices: @Array<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<i33>, index: usize) -> Array<usize> {
        unravel_index(index, *self.shape)
    }

    fn broadcast_index_mapping(self: @Tensor<i33>, indices: @Array<usize>) -> usize {
        broadcast_index_mapping(*self.shape, indices)
    }

    fn add(self: @Tensor<i33>, other: @Tensor<i33>) -> Tensor<i33> {
        i33_add_tensor(self, other)
    }

    fn sub(self: @Tensor<i33>, other: @Tensor<i33>) -> Tensor<i33> {
        i33_sub_tensor(self, other)
    }

    fn mul(self: @Tensor<i33>, other: @Tensor<i33>) -> Tensor<i33> {
        i33_mul_tensor(self, other)
    }

    fn div(self: @Tensor<i33>, other: @Tensor<i33>) -> Tensor<i33> {
        i33_div_tensor(self, other)
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

    let i = self.broadcast_index_mapping(@indices_self);
    let j = other.broadcast_index_mapping(@indices_other);

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

    let i = self.broadcast_index_mapping(@indices_self);
    let j = other.broadcast_index_mapping(@indices_other);

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

    let i = self.broadcast_index_mapping(@indices_self);
    let j = other.broadcast_index_mapping(@indices_other);

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

    let i = self.broadcast_index_mapping(@indices_self);
    let j = other.broadcast_index_mapping(@indices_other);

    result.append(*(*self.data).at(i) / *(*other.data).at(j));
    __i33_div_tensor(self, other, ref result, n + 1_usize);
}

