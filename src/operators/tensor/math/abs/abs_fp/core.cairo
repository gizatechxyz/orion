use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::math::abs::abs_fp::fp8x23;
use orion::operators::tensor::math::abs::abs_fp::fp16x16;


/// Cf: TensorTrait::abs docstring
fn abs(z: @Tensor<FixedType>) -> Option<Tensor<FixedType>> {
    match *z.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::abs(z)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::abs(z)),
            },
            Option::None(_) => Option::Some(fp16x16::abs(z)),
        },
        Option::None(_) => Option::Some(fp16x16::abs(z)),
    }
}
