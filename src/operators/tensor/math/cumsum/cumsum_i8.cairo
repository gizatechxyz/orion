use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::cumsum::helpers::{cumsum_forward, cumsum_reverse};

/// Cf: TensorTrait::cumsum docstring
fn cumsum(
    self: @Tensor<i8>, 
    axis: usize,
    exclusive: Option<bool>, 
    reverse: Option<bool> 
 ) -> Tensor<i8> {

    let reverse = match reverse {
        Option::Some(val) => val,
        Option::None(_) => false
    };

    if reverse {
        cumsum_reverse(self, axis, exclusive, IntegerTrait::new(0,false))
    } else {
        cumsum_forward(self, axis, exclusive, IntegerTrait::new(0,false))
    }
}
