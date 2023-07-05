use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::math::arithmetic::arithmetic_i32::{saturated_add, saturated_div};
use orion::utils::saturate;

#[derive(Drop)]
enum Yscale {
    ElementWise: i32,
    PerAxis: Tensor<i32>
}

#[derive(Drop)]
enum ZeroPoint {
    ElementWise: i32,
    PerAxis: Tensor<i32>
}

fn quantize_linear(x: @Tensor<i32>, y_scale: Yscale, y_zero_point: ZeroPoint) -> Tensor::<i32> {
    let y = match y_scale {
        Yscale::ElementWise(scale) => match y_zero_point {
            ZeroPoint::ElementWise(zero) => Option::Some(quantize_element_wise(x, scale, zero)),
            ZeroPoint::PerAxis(zero) => Option::None(())
        },
        Yscale::PerAxis(scale_tensor) => match y_zero_point {
            ZeroPoint::ElementWise(zero) => Option::None(()),
            ZeroPoint::PerAxis(zero_tensor) => Option::Some(
                quantize_per_axis(x, @scale_tensor, @zero_tensor)
            )
        },
    };

    assert(y.is_some(), 'Mismatch w/ Yscale & ZeroPoint');
    return y.unwrap();
}

fn quantize_per_axis(
    x: @Tensor<i32>, y_scale: @Tensor<i32>, y_zero_point: @Tensor<i32>
) -> Tensor::<i32> {
    let mut result_data = ArrayTrait::<i32>::new();

    let min = IntegerTrait::new(128, true);
    let max = IntegerTrait::new(127, false);

    saturated_add(@saturated_div(x, y_scale, min, max), y_zero_point, min, max)
}

fn quantize_element_wise(x: @Tensor::<i32>, y_scale: i32, y_zero_point: i32) -> Tensor::<i32> {
    let mut result_data = ArrayTrait::<i32>::new();
    let mut data = *x.data;

    loop {
        let quantized = quantize(y_scale, y_zero_point, *data.pop_front().unwrap());
        result_data.append(quantized);

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::new(*x.shape, result_data.span(), *x.extra);
}

fn quantize(x: i32, y_scale: i32, y_zero_point: i32) -> i32 {
    saturate(
        IntegerTrait::new(128, true), IntegerTrait::new(127, false), ((x / y_scale) + y_zero_point)
    )
}
