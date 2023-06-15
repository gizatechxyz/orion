use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::numbers::signed_integer::i32::i32;

use orion::numbers::fixed_point::implementations::impl_16x16;
use orion::utils::check_gas;

/// Cf: TensorTrait::acosh docstring
fn acosh(self: @Tensor<u32>) -> Tensor<FixedType> {
    let mut result = ArrayTrait::new();
    let mut data = *self.data;

    loop {
        check_gas();

        let ele = *data.pop_front().unwrap();

        if FixedTrait::from_unscaled_felt(ele.into()) < FixedTrait::new_unscaled(1, false) {
            let mut data = ArrayTrait::new();
            data.append('inputs must be >= to 1');
            panic(data);
        }
        else {  
            let x = FixedTrait::from_unscaled_felt(ele.into());
            let x_sq_minus_one = (x * x) - FixedTrait::from_unscaled_felt(1);
            let sqrt = x_sq_minus_one.sqrt();
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