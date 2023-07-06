use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor, ExtraParams};
use orion::performance::functional::quantize_linear::quantize_linear_fp::fp_i32::fp8x23;
use orion::performance::functional::quantize_linear::quantize_linear_fp::fp_i32::fp16x16;
use orion::numbers::signed_integer::i32::i32;

/// Cf: PerfomanceTrait::quantize_linear docstring
fn quantize_linear(
    x: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
) -> Option<Tensor<i32>> {
    match *x.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(
                    fp8x23::quantize_linear(x, y_scale, y_zero_point)
                ),
                FixedImpl::FP16x16(()) => Option::Some(
                    fp16x16::quantize_linear(x, y_scale, y_zero_point)
                ),
            },
            Option::None(_) => Option::Some(fp16x16::quantize_linear(x, y_scale, y_zero_point)),
        },
        Option::None(_) => Option::Some(fp16x16::quantize_linear(x, y_scale, y_zero_point)),
    }
}
