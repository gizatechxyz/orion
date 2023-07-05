use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::implementations::impl_16x16::{
    FP16x16Impl, FP16x16PartialOrd, FP16x16Div, FP16x16Add
};
use orion::operators::tensor::math::arithmetic::arithmetic_fp::fp16x16::{
    saturated_add, saturated_div
};
use orion::performance::functional::quantize_linear::quantize_linear_fp::core::{Yscale, ZeroPoint};
use orion::utils::saturate;


fn quantize_linear(
    x: @Tensor<FixedType>, y_scale: Yscale, y_zero_point: ZeroPoint
) -> Tensor::<FixedType> {
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
    x: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
) -> Tensor::<FixedType> {
    let mut result_data = ArrayTrait::<FixedType>::new();

    let min = FixedTrait::new_unscaled(128, true);
    let max = FixedTrait::new_unscaled(127, false);

    saturated_add(@saturated_div(x, y_scale, min, max), y_zero_point, min, max)
}

fn quantize_element_wise(
    x: @Tensor::<FixedType>, y_scale: FixedType, y_zero_point: FixedType
) -> Tensor::<FixedType> {
    let mut result_data = ArrayTrait::<FixedType>::new();
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

fn quantize(x: FixedType, y_scale: FixedType, y_zero_point: FixedType) -> FixedType {
    saturate(
        FixedTrait::new_unscaled(128, true),
        FixedTrait::new_unscaled(127, false),
        ((x / y_scale) + y_zero_point)
    )
}
