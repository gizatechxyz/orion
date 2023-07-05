use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::math::arithmetic::arithmetic_u32::{saturated_add, saturated_div};
use orion::utils::saturate;

#[derive(Drop)]
enum Yscale {
    ElementWise: u32,
    PerAxis: Tensor<u32>
}

#[derive(Drop)]
enum ZeroPoint {
    ElementWise: u32,
    PerAxis: Tensor<u32>
}

fn quantize_linear(x: @Tensor<u32>, y_scale: Yscale, y_zero_point: ZeroPoint) -> Tensor::<u32> {
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
    x: @Tensor<u32>, y_scale: @Tensor<u32>, y_zero_point: @Tensor<u32>
) -> Tensor::<u32> {
    let mut result_data = ArrayTrait::<u32>::new();

    let min = 0;
    let max = 255;

    saturated_add(@saturated_div(x, y_scale, min, max), y_zero_point, min, max)
}

fn quantize_element_wise(x: @Tensor::<u32>, y_scale: u32, y_zero_point: u32) -> Tensor::<u32> {
    let mut result_data = ArrayTrait::<u32>::new();
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

fn quantize(x: u32, y_scale: u32, y_zero_point: u32) -> u32 {
    saturate(0, 255, ((x / y_scale) + y_zero_point))
}
