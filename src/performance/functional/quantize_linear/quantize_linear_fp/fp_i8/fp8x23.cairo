use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::TryInto;

use orion::numbers::signed_integer::i8::i8;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp, FixedTypeTensorDiv};
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::numbers::fixed_point::implementations::fp8x23::core::{
    FP8x23Impl, FP8x23PartialOrd, FP8x23Div, FP8x23Add, FP8x23TryIntoI8
};
use orion::operators::tensor::math::arithmetic::arithmetic_fp::fp8x23::{
    saturated_add_i8, saturated_div
};
use orion::operators::tensor::helpers::check_compatibility;
use orion::utils::{saturate};

/// Cf: PerfomanceTrait::quantize_linear docstring
fn quantize_linear(
    x: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
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
    x: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
) -> Tensor::<i8> {
    saturated_add_i8(@(*x / *y_scale), y_zero_point)
}

fn quantize_element_wise(
    mut x: Tensor::<FixedType>, y_scale: FixedType, y_zero_point: FixedType
) -> Tensor::<i8> {
    let mut result_data = ArrayTrait::<i8>::new();

    loop {
        match x.data.pop_front() {
            Option::Some(item) => {
                let quantized = quantize(*item, y_scale, y_zero_point);
                result_data.append(quantized.try_into().unwrap());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(x.shape, result_data.span(), x.extra);
}


fn quantize(x: FixedType, y_scale: FixedType, y_zero_point: FixedType) -> FixedType {
    saturate(
        FixedTrait::new_unscaled(128, true),
        FixedTrait::new_unscaled(127, false),
        ((x / y_scale) + y_zero_point)
    )
}
