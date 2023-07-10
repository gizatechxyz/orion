use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_u32::{
    Tensor_u32, u32TensorSub, u32TensorMul
};
use orion::operators::tensor::math::arithmetic::arithmetic_u32::{saturated_add, saturated_div};
use orion::operators::tensor::helpers::check_compatibility;
use orion::utils::saturate;

/// Cf: PerfomanceTrait::dequantize_linear docstring
fn dequantize_linear(
    x: @Tensor<u32>, x_scale: @Tensor<u32>, x_zero_point: @Tensor<u32>
) -> Tensor::<u32> {
    if (*x_scale.data).len() == 1 && (*x_zero_point.data).len() == 1 {
        dequantize_element_wise(x, *x_scale.data[0], *x_zero_point.data[0])
    } else {
        check_compatibility(*x.shape, *x_scale.shape);
        check_compatibility(*x.shape, *x_zero_point.shape);
        check_compatibility(*x_scale.shape, *x_zero_point.shape);
        dequantize_per_axis(x, x_scale, x_zero_point)
    }
}

fn dequantize_per_axis(
    x: @Tensor<u32>, x_scale: @Tensor<u32>, x_zero_point: @Tensor<u32>
) -> Tensor::<u32> {
    let mut result_data = ArrayTrait::<u32>::new();

    (*x - *x_zero_point) * *x_scale
}

fn dequantize_element_wise(x: @Tensor::<u32>, x_scale: u32, x_zero_point: u32) -> Tensor::<u32> {
    let mut result_data = ArrayTrait::<u32>::new();
    let mut data = *x.data;

    loop {
        let dequantized = dequantize(*data.pop_front().unwrap(), x_scale, x_zero_point);
        result_data.append(dequantized);

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::new(*x.shape, result_data.span(), *x.extra);
}

fn dequantize(x: u32, x_scale: u32, x_zero_point: u32) -> u32 {
    (x - x_zero_point) * x_scale
}
