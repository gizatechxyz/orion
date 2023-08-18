use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23::{
    FP8x23Impl, FP8x23Add, FP8x23Sub, FP8x23AddEq
};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::concat::helpers::{ concat_helper};


/// Cf: TensorTrait::cumsum docstring
/// Cf: TensorTrait::cumsum docstring
fn concat(
   tensors: Span<Tensor<FixedType>>, axis: usize,  
) -> Tensor<FixedType> {

    concat_helper(tensors, axis)
}
