use core::debug::PrintTrait;

use orion::operators::tensor::core::{Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::nn::functional::leaky_relu::leaky_relu_u32::fp16x16;
use orion::operators::nn::functional::leaky_relu::leaky_relu_u32::fp8x23;

/// Cf: NNTrait::leaky_relu docstring
fn leaky_relu_u32(
    z: @Tensor<u32>, alpha: @FixedType, threshold: u32
) -> Option::<Tensor<FixedType>> {
    match *z.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::leaky_relu(z, alpha, threshold)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::leaky_relu(z, alpha, threshold)),
            },
            Option::None(_) => Option::Some(fp16x16::leaky_relu(z, alpha, threshold)),
        },
        Option::None(_) => Option::Some(fp16x16::leaky_relu(z, alpha, threshold)),
    }
}

