use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::cumsum::helpers::{cumsum_forward, cumsum_reverse};

/// Cf: TensorTrait::cumsum docstring
fn cumsum(
    self: @Tensor<u32>, 
    axis: usize,
    exclusive: Option<bool>, 
    reverse: Option<bool>, 
 ) -> Tensor<u32> {

    let reverse = match reverse {
        Option::Some(val) => val,
        Option::None(_) => false
    };

    if reverse {
        cumsum_reverse(self, axis, exclusive, 0)
    } else {
        cumsum_forward(self, axis, exclusive, 0)
    }
}
