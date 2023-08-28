use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16Impl, FP16x16Add, FP16x16Sub, FP16x16AddEq
};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::concat::helpers::{concat_helper};


/// Cf: TensorTrait::cumsum docstring
fn concat(
   tensors: Span<Tensor<FixedType>>, axis: usize,  
) -> Tensor<FixedType> {

    concat_helper(tensors, axis)
}
