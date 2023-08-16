use core::debug::PrintTrait;

use orion::operators::tensor::core::{Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::nn::functional::softplus::softplus_fp::fp16x16;
use orion::operators::nn::functional::softplus::softplus_fp::fp8x23;

/// Cf: NNTrait::softplus docstring
fn softplus_fp(z: @Tensor<FixedType>) -> Option::<Tensor<FixedType>> {
    match *z.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::softplus(z)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::softplus(z)),
            },
            Option::None(_) => Option::Some(fp16x16::softplus(z)),
        },
        Option::None(_) => Option::Some(fp16x16::softplus(z)),
    }
}

