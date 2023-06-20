use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::Tensor;
use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
use orion::operators::tensor::linalg::transpose::transpose_fp::fp8x23;
use orion::operators::tensor::linalg::transpose::transpose_fp::fp16x16;
use orion::utils::check_gas;

/// Cf: TensorTrait::transpose docstring
fn transpose(self: @Tensor<FixedType>, axes: Span<usize>) -> Option<Tensor<FixedType>> {
    match *self.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::transpose(self, axes)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::transpose(self, axes)),
            },
            Option::None(_) => Option::Some(fp16x16::transpose(self, axes)),
        },
        Option::None(_) => Option::Some(fp16x16::transpose(self, axes)),
    }
}
