use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::cumsum::helpers::{cumsum_forward, cumsum_reverse};
use orion::operators::tensor::math::concat::helpers::{concat_helper};


/// Cf: TensorTrait::cumsum docstring
fn concat_u32(
   tensors: Span<Tensor<u32>>, axis: usize,  
) -> Tensor<u32> {

    concat_helper(tensors, axis)
}