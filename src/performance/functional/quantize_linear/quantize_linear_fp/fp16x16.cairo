use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp, FixedTypeTensorDiv};
use orion::numbers::fixed_point::implementations::impl_16x16::{
    FP16x16Impl, FP16x16PartialOrd, FP16x16Div, FP16x16Add
};
use orion::operators::tensor::math::arithmetic::arithmetic_fp::fp16x16::{
    saturated_add, saturated_div
};
use orion::operators::tensor::helpers::check_compatibility;
use orion::utils::saturate;

fn quantize_linear(
    x: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
) -> Tensor::<FixedType> {
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
    x: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
) -> Tensor::<FixedType> {
    let mut result_data = ArrayTrait::<FixedType>::new();

    let min = FixedTrait::new_unscaled(128, true);
    let max = FixedTrait::new_unscaled(127, false);

    saturated_add(@(*x / *y_scale), y_zero_point, min, max)
}

fn quantize_element_wise(
    x: @Tensor::<FixedType>, y_scale: FixedType, y_zero_point: FixedType
) -> Tensor::<FixedType> {
    let mut result_data = ArrayTrait::<FixedType>::new();
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

fn quantize(x: FixedType, y_scale: FixedType, y_zero_point: FixedType) -> FixedType {
    saturate(
        FixedTrait::new_unscaled(128, true),
        FixedTrait::new_unscaled(127, false),
        ((x / y_scale) + y_zero_point)
    )
}
