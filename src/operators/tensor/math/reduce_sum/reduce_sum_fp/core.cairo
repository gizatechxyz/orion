use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor};
use orion::operators::tensor::math::reduce_sum::reduce_sum_fp::fp8x23;
use orion::operators::tensor::math::reduce_sum::reduce_sum_fp::fp16x16;


/// Cf: TensorTrait::reduce_sum docstring
fn reduce_sum(self: @Tensor<FixedType>, axis: usize, keepdims: bool) -> Option<Tensor<FixedType>> {
    match *self.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::reduce_sum(self, axis, keepdims)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::reduce_sum(self, axis, keepdims)),
            },
            Option::None(_) => Option::Some(fp16x16::reduce_sum(self, axis, keepdims)),
        },
        Option::None(_) => Option::Some(fp16x16::reduce_sum(self, axis, keepdims)),
    }
}

