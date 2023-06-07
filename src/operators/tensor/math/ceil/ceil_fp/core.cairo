use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_16x16;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::math::ceil::ceil_fp::fp8x23;
use orion::operators::tensor::math::ceil::ceil_fp::fp16x16;
use orion::utils::check_gas;

/// Cf: TensorTrait::ceil docstring
fn ceil(z: @Tensor<FixedType>) -> Option<Tensor<FixedType>> {
    match *z.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::ceil(z)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::ceil(z)),
            },
            Option::None(_) => Option::Some(fp16x16::ceil(z)),
        },
        Option::None(_) => Option::Some(fp16x16::ceil(z)),
    }
}
