use array::{ArrayTrait};
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;

use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::utils::check_gas;

/// Cf: TensorTrait::acosh docstring
fn acosh(self: @Tensor<FixedType>) -> Tensor<FixedType> {
   
    let mut result = ArrayTrait::<FixedType>::new();
    let mut data = *self.data;

    loop {
        check_gas();
        let ele = *data.pop_front().unwrap();
        if ele.sign == true | (FixedTrait::new_unscaled(ele.mag, false) < FixedTrait::new_unscaled(1, false)) {
            let mut data = ArrayTrait::new();
            data.append('inputs must be >= to 1');
            panic(data);
        }
        else {
            let x = ele; 
            let x_sq_plus_one = (x * x) - FixedTrait::new_unscaled(1, false);
            let sqrt = x_sq_plus_one.sqrt();
            let sqrt_plus_x = sqrt + x;
            let answer = sqrt_plus_x.ln();
            result.append(answer);
        }
        if (data.len() == 0) {
          break ();
        };
        
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span(), *self.extra);
}
