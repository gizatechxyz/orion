use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::fp8x23::core::{
    FP8x23Impl, FP8x23Add, FP8x23Sub, FP8x23AddEq
};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::cumsum::helpers::{cumsum_forward, cumsum_reverse};


/// Cf: TensorTrait::cumsum docstring
fn cumsum(
    self: @Tensor<FixedType>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
) -> Tensor<FixedType> {
    let reverse = match reverse {
        Option::Some(val) => val,
        Option::None(_) => false
    };

    if reverse {
        cumsum_reverse(self, axis, exclusive, FixedTrait::new(0, false))
    } else {
        cumsum_forward(self, axis, exclusive, FixedTrait::new(0, false))
    }
}
