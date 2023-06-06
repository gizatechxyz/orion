use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor, ExtraParams};
use orion::performance::functional::quantization::quant_fp::fp8x23;
use orion::performance::functional::quantization::quant_fp::fp16x16;

/// Cf: PerfomanceTrait::quantize_linear docstring
fn quantize_tensor(tensor: @Tensor::<FixedType>) -> Option<Tensor<FixedType>> {
    match *tensor.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::quantize_tensor(tensor)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::quantize_tensor(tensor)),
            },
            Option::None(_) => Option::Some(fp16x16::quantize_tensor(tensor)),
        },
        Option::None(_) => Option::Some(fp16x16::quantize_tensor(tensor)),
    }
}

fn symetric_quant(
    min_val: FixedType, max_val: FixedType, data: FixedType, extra: Option<ExtraParams>
) -> Option<FixedType> {
    match extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(
                    fp8x23::symetric_quant(min_val, max_val, data)
                ),
                FixedImpl::FP16x16(()) => Option::Some(
                    fp16x16::symetric_quant(min_val, max_val, data)
                ),
            },
            Option::None(_) => Option::Some(fp16x16::symetric_quant(min_val, max_val, data)),
        },
        Option::None(_) => Option::Some(fp16x16::symetric_quant(min_val, max_val, data)),
    }
}
