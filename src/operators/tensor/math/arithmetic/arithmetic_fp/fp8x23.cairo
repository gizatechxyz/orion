use core::traits::Into;
use core::option::OptionTrait;
use array::ArrayTrait;
use array::SpanTrait;
use traits::{TryInto};

use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::impl_8x23::{
    FP8x23Add, FP8x23Sub, FP8x23Mul, FP8x23Div, FP8x23PartialOrd, FP8x23TryIntoI32, FP8x23TryIntoU32
};
use orion::operators::tensor::helpers::broadcast_shape;
use orion::operators::tensor::core::{Tensor, TensorTrait, unravel_index};
use orion::operators::tensor::helpers::{broadcast_index_mapping, len_from_shape, };
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::utils::check_gas;
use orion::utils::{saturate};

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
fn add(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] + *(*other.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise addition of two `Tensor<FixedType>` instances with broadcasting and saturation.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<FixedType>` instance representing the result of the element-wise addition with broadcasting and saturation.
fn saturated_add(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<FixedType> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                saturate(min, max, *(*self.data)[indices_self] + *(*other.data)[indices_other])
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise addition of two `Tensor<FixedType>` instances with broadcasting and saturation, and convert to i32 tensor.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<i32>` instance representing the result of the element-wise addition with broadcasting and saturation.
fn saturated_add_to_i32(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<i32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                (saturate(min, max, *(*self.data)[indices_self] + *(*other.data)[indices_other]))
                    .try_into()
                    .unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<i32>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise addition of two `Tensor<FixedType>` instances with broadcasting and saturation, and convert to u32 tensor.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the element-wise addition with broadcasting and saturation.
fn saturated_add_to_u32(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<u32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                (
                    saturate(min, max, *(*self.data)[indices_self] + *(*other.data)[indices_other])
                ).try_into().unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<u32>::new(broadcasted_shape, result.span(), *self.extra);
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
fn sub(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] - *(*other.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise substraction of two `Tensor<FixedType>` instances with broadcasting and saturation.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<FixedType>` instance representing the result of the element-wise substraction with broadcasting and saturation.
fn saturated_sub(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<FixedType> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                saturate(min, max, *(*self.data)[indices_self] - *(*other.data)[indices_other])
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise substraction of two `Tensor<FixedType>` instances with broadcasting and saturation, and convert to i32 tensor.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<i32>` instance representing the result of the element-wise substraction with broadcasting and saturation.
fn saturated_sub_to_i32(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<i32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                (saturate(min, max, *(*self.data)[indices_self] - *(*other.data)[indices_other]))
                    .try_into()
                    .unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<i32>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise substraction of two `Tensor<FixedType>` instances with broadcasting and saturation, and convert to u32 tensor.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the element-wise substraction with broadcasting and saturation.
fn saturated_sub_to_u32(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<u32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                (
                    saturate(min, max, *(*self.data)[indices_self] - *(*other.data)[indices_other])
                ).try_into().unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<u32>::new(broadcasted_shape, result.span(), *self.extra);
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
fn mul(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] * *(*other.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise multiplication of two `Tensor<FixedType>` instances with broadcasting and saturation.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<FixedType>` instance representing the result of the element-wise multiplication with broadcasting and saturation.
fn saturated_mul(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<FixedType> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                saturate(min, max, *(*self.data)[indices_self] * *(*other.data)[indices_other])
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise multiplication of two `Tensor<FixedType>` instances with broadcasting and saturation, and convert to i32 tensor.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<i32>` instance representing the result of the element-wise multiplication with broadcasting and saturation.
fn saturated_mul_to_i32(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<i32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                (saturate(min, max, *(*self.data)[indices_self] * *(*other.data)[indices_other]))
                    .try_into()
                    .unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<i32>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise multiplication of two `Tensor<FixedType>` instances with broadcasting and saturation, and convert to u32 tensor.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the element-wise multiplication with broadcasting and saturation.
fn saturated_mul_to_u32(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<u32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                (
                    saturate(min, max, *(*self.data)[indices_self] * *(*other.data)[indices_other])
                ).try_into().unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<u32>::new(broadcasted_shape, result.span(), *self.extra);
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
fn div(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] / *(*other.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise division of two `Tensor<FixedType>` instances with broadcasting and saturation.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<FixedType>` instance representing the result of the element-wise division with broadcasting and saturation.
fn saturated_div(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<FixedType> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                saturate(min, max, *(*self.data)[indices_self] / *(*other.data)[indices_other])
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise division of two `Tensor<FixedType>` instances with broadcasting and saturation, and convert to i32 tensor.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<i32>` instance representing the result of the element-wise division with broadcasting and saturation.
fn saturated_div_to_i32(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<i32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                (saturate(min, max, *(*self.data)[indices_self] / *(*other.data)[indices_other]))
                    .try_into()
                    .unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<i32>::new(broadcasted_shape, result.span(), *self.extra);
}

/// Performs element-wise division of two `Tensor<FixedType>` instances with broadcasting and saturation, and convert to u32 tensor.
///
/// # Arguments
/// * `self` - The first tensor.
/// * `other` - The second tensor.
/// * `min` - The minimum value for saturation.
/// * `max` - The maximum value for saturation.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if the gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the element-wise division with broadcasting and saturation.
fn saturated_div_to_u32(
    self: @Tensor<FixedType>, other: @Tensor<FixedType>, min: FixedType, max: FixedType
) -> Tensor<u32> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        check_gas();

        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                (
                    saturate(min, max, *(*self.data)[indices_self] / *(*other.data)[indices_other])
                ).try_into().unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<u32>::new(broadcasted_shape, result.span(), *self.extra);
}
