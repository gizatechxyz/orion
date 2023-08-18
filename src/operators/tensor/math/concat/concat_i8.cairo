use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::core::Tensor;

use orion::operators::tensor::math::concat::helpers::{ concat_helper};


/// Cf: TensorTrait::cumsum docstring
/// Cf: TensorTrait::cumsum docstring
fn concat_i8(
   tensors: Span<Tensor<i8>>, axis: usize,  
) -> Tensor<i8> {

    concat_helper(tensors, axis)
}