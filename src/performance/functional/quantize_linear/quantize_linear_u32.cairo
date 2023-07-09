use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32, u32TensorDiv};
use orion::operators::tensor::math::arithmetic::arithmetic_u32::{saturated_add, saturated_div};
use orion::operators::tensor::helpers::check_compatibility;
use orion::utils::saturate;

/// Cf: PerfomanceTrait::quantize_linear docstring
fn quantize_linear(
    x: @Tensor<u32>, y_scale: @Tensor<u32>, y_zero_point: @Tensor<u32>
) -> Tensor::<u32> {
    if (*y_scale.data).len() == 1 && (*y_zero_point.data).len() == 1 {
        quantize_element_wise(x, *y_scale.data[0], *y_zero_point.data[0])
    } else {
        check_compatibility(*x.shape, *y_scale.shape);
        check_compatibility(*x.shape, *y_zero_point.shape);
        check_compatibility(*y_scale.shape, *y_zero_point.shape);
        quantize_per_axis(x, y_scale, y_zero_point)
    }
}

fn quantize_per_axis(
    x: @Tensor<u32>, y_scale: @Tensor<u32>, y_zero_point: @Tensor<u32>
) -> Tensor::<u32> {
    let mut result_data = ArrayTrait::<u32>::new();

    let min = 0;
    let max = 255;

    saturated_add(@(*x / *y_scale), y_zero_point, min, max)
}

fn quantize_element_wise(x: @Tensor::<u32>, y_scale: u32, y_zero_point: u32) -> Tensor::<u32> {
    let mut result_data = ArrayTrait::<u32>::new();
    let mut data = *x.data;

    loop {
        let quantized = quantize(*data.pop_front().unwrap(), y_scale, y_zero_point);
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
