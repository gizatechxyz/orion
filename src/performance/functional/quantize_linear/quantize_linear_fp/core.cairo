use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor, ExtraParams};
use orion::performance::functional::quantize_linear::quantize_linear_fp::fp8x23;
use orion::performance::functional::quantize_linear::quantize_linear_fp::fp16x16;


#[derive(Drop)]
enum Yscale {
    ElementWise: FixedType,
    PerAxis: Tensor<FixedType>
}

#[derive(Drop)]
enum ZeroPoint {
    ElementWise: FixedType,
    PerAxis: Tensor<FixedType>
}

/// Cf: PerfomanceTrait::quantize_linear docstring
fn quantize_linear(
    tensor: @Tensor::<FixedType>, y_scale: Yscale, y_zero_point: ZeroPoint
) -> Option<Tensor<FixedType>> {
    match *tensor.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(
                    fp8x23::quantize_linear(tensor, y_scale, y_zero_point)
                ),
                FixedImpl::FP16x16(()) => Option::Some(
                    fp16x16::quantize_linear(tensor, y_scale, y_zero_point)
                ),
            },
            Option::None(_) => Option::Some(
                fp16x16::quantize_linear(tensor, y_scale, y_zero_point)
            ),
        },
        Option::None(_) => Option::Some(fp16x16::quantize_linear(tensor, y_scale, y_zero_point)),
    }
}
