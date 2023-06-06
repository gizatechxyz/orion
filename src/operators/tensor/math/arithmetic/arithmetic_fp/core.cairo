use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor};
use orion::operators::tensor::math::arithmetic::arithmetic_fp::fp8x23;
use orion::operators::tensor::math::arithmetic::arithmetic_fp::fp16x16;

fn add(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Option<Tensor<FixedType>> {
    match *self.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::add(self, other)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::add(self, other)),
            },
            Option::None(_) => Option::Some(fp16x16::add(self, other)),
        },
        Option::None(_) => Option::Some(fp16x16::add(self, other)),
    }
}

fn sub(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Option<Tensor<FixedType>> {
    match *self.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::sub(self, other)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::sub(self, other)),
            },
            Option::None(_) => Option::Some(fp16x16::sub(self, other)),
        },
        Option::None(_) => Option::Some(fp16x16::sub(self, other)),
    }
}

fn mul(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Option<Tensor<FixedType>> {
    match *self.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::mul(self, other)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::mul(self, other)),
            },
            Option::None(_) => Option::Some(fp16x16::mul(self, other)),
        },
        Option::None(_) => Option::Some(fp16x16::mul(self, other)),
    }
}

fn div(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Option<Tensor<FixedType>> {
    match *self.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::div(self, other)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::div(self, other)),
            },
            Option::None(_) => Option::Some(fp16x16::div(self, other)),
        },
        Option::None(_) => Option::Some(fp16x16::div(self, other)),
    }
}
