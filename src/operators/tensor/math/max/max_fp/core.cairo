use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::ExtraParams;
use orion::operators::tensor::math::max::max_fp::fp8x23;
use orion::operators::tensor::math::max::max_fp::fp16x16;

/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<FixedType>, extra: Option<ExtraParams>) -> Option<FixedType> {
    match extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::max_in_tensor(vec)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::max_in_tensor(vec)),
            },
            Option::None(_) => Option::Some(fp16x16::max_in_tensor(vec)),
        },
        Option::None(_) => Option::Some(fp16x16::max_in_tensor(vec)),
    }
}
