use core::debug::PrintTrait;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::TryInto;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32, i8::i8};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32, i32TensorDiv, };
use orion::operators::tensor::implementations::impl_tensor_i8::{Tensor_i8, i8TensorDiv, };
use orion::operators::tensor::math::arithmetic::arithmetic_i32::{saturated_add_i8};
use orion::operators::tensor::helpers::check_compatibility;
use orion::utils::saturate;

/// Cf: PerfomanceTrait::quantize_linear docstring
fn quantize_linear(
    x: @Tensor<i32>, y_scale: @Tensor<i32>, y_zero_point: @Tensor<i32>
) -> Tensor::<i8> {
    if (*y_scale.data).len() == 1 && (*y_zero_point.data).len() == 1 {
        quantize_element_wise(*x, *y_scale.data[0], *y_zero_point.data[0])
    } else {
        check_compatibility(*x.shape, *y_scale.shape);
        check_compatibility(*x.shape, *y_zero_point.shape);
        check_compatibility(*y_scale.shape, *y_zero_point.shape);
        quantize_per_axis(x, y_scale, y_zero_point)
    }
}

fn quantize_per_axis(
    x: @Tensor<i32>, y_scale: @Tensor<i32>, y_zero_point: @Tensor<i32>
) -> Tensor::<i8> {
    saturated_add_i8(@(*x / *y_scale), y_zero_point)
}

fn quantize_element_wise(mut x: Tensor::<i32>, y_scale: i32, y_zero_point: i32) -> Tensor::<i8> {
    let mut result_data = ArrayTrait::<i8>::new();

    loop {
        match x.data.pop_front() {
            Option::Some(item) => {
                let quantized = quantize(*item, y_scale, y_zero_point);
                result_data.append(quantized);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(x.shape, result_data.span(), x.extra);
}

fn quantize(x: i32, y_scale: i32, y_zero_point: i32) -> i8 {
    saturate(
        IntegerTrait::new(128, true), IntegerTrait::new(127, false), ((x / y_scale) + y_zero_point)
    )
        .try_into()
        .unwrap()
}
