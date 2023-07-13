use orion::numbers::signed_integer::i8::i8;
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor};
use orion::performance::functional::dequantize_linear::dequantize_linear_fp::fp8x23;
use orion::performance::functional::dequantize_linear::dequantize_linear_fp::fp16x16;

/// Cf: PerfomanceTrait::dequantize_linear docstring
fn dequantize_linear(
    x: @Tensor<i8>, x_scale: @Tensor<FixedType>, x_zero_point: @Tensor<FixedType>
) -> Option<Tensor<FixedType>> {
    match *x.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(
                    fp8x23::dequantize_linear(x, x_scale, x_zero_point)
                ),
                FixedImpl::FP16x16(()) => Option::Some(
                    fp16x16::dequantize_linear(x, x_scale, x_zero_point)
                ),
            },
            Option::None(_) => Option::Some(fp16x16::dequantize_linear(x, x_scale, x_zero_point)),
        },
        Option::None(_) => Option::Some(fp16x16::dequantize_linear(x, x_scale, x_zero_point)),
    }
}
