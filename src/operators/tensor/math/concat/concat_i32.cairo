use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::concat::helpers::{ concat_helper};


/// Cf: TensorTrait::cumsum docstring
/// Cf: TensorTrait::cumsum docstring
fn concat_i32(
   tensors: Span<Tensor<i32>>, axis: usize,  
) -> Tensor<i32> {

    concat_helper(tensors, axis)
}
