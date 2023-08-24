use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32, i8::i8};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i32::{
    Tensor_i32, i32TensorSub, i32TensorMul
};
use orion::operators::tensor::implementations::impl_tensor_i8::TensorI8IntoTensorI32;
use orion::operators::tensor::helpers::check_compatibility;
use orion::utils::saturate;

/// Cf: PerfomanceTrait::dequantize_linear docstring
fn dequantize_linear(
    x: @Tensor<i8>, x_scale: @Tensor<i32>, x_zero_point: @Tensor<i32>
) -> Tensor::<i32> {
    if (*x_scale.data).len() == 1 && (*x_zero_point.data).len() == 1 {
        dequantize_element_wise(*x, *x_scale.data[0], *x_zero_point.data[0])
    } else {
        check_compatibility(*x.shape, *x_scale.shape);
        check_compatibility(*x.shape, *x_zero_point.shape);
        check_compatibility(*x_scale.shape, *x_zero_point.shape);
        dequantize_per_axis(@(*x).into(), x_scale, x_zero_point)
    }
}

fn dequantize_per_axis(
    x: @Tensor<i32>, x_scale: @Tensor<i32>, x_zero_point: @Tensor<i32>
) -> Tensor::<i32> {
    let mut result_data = ArrayTrait::<i32>::new();

    (*x - *x_zero_point) * *x_scale
}

fn dequantize_element_wise(mut x: Tensor::<i8>, x_scale: i32, x_zero_point: i32) -> Tensor::<i32> {
    let mut result_data = ArrayTrait::<i32>::new();

    loop {
        match x.data.pop_front() {
            Option::Some(item) => {
                let dequantized = dequantize(*item, x_scale, x_zero_point);
                result_data.append(dequantized);
            },
            Option::None(_) => {
                break;
            }
        };
    };


    return TensorTrait::new(x.shape, result_data.span(), x.extra);
}

fn dequantize(x: i8, x_scale: i32, x_zero_point: i32) -> i32 {
    (x.into() - x_zero_point) * x_scale
}
